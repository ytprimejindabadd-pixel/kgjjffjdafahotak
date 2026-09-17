import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Optional
import threading
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel
import pyarrow.parquet as pq
import pandas as pd
from io import BytesIO

# ── Config ──────────────────────────────────────────────────────────────────
HF_DATASET_URL = os.environ.get(
    "ICMR_HF_DATASET_URL",
    "https://huggingface.co/datasets/rehuuuu/icrm-hitek-fulldb/resolve/main",
).rstrip("/")

PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "2"))
TIMEOUT = int(os.environ.get("ICMR_TIMEOUT", "60"))
DUPLICATE_CAP = 2

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

# ── AUTO-DISCOVER PARQUET FILES ─────────────────────────────────────────────
# Instead of hardcoding 7 files, we auto-discover what exists
_parquet_cache: dict[str, list[str]] = {}
_parquet_lock = threading.Lock()

async def discover_parquet_files(client: httpx.AsyncClient) -> dict[str, list[str]]:
    """Auto-discover which parquet files exist on HF."""
    global _parquet_cache
    if _parquet_cache:
        return _parquet_cache
    
    found = {"phone": [], "aadhar": [], "other": []}
    
    # Try HF API to list files
    try:
        api_url = HF_DATASET_URL.replace("/resolve/main", "").replace(
            "https://huggingface.co/datasets/", "https://huggingface.co/api/datasets/"
        )
        r = await client.get(api_url, timeout=30)
        if r.status_code == 200:
            info = r.json()
            siblings = info.get("siblings", [])
            for s in siblings:
                fname = s.get("rfilename", "")
                if fname.endswith(".parquet"):
                    url = f"{HF_DATASET_URL}/{fname}"
                    low = fname.lower()
                    if "phone" in low or "mobile" in low or "contact" in low:
                        found["phone"].append(url)
                    elif "aadhar" in low or "aadhaar" in low:
                        found["aadhar"].append(url)
                    else:
                        found["other"].append(url)
    except Exception as e:
        print(f"⚠️ HF API discovery failed: {e}")
    
    # Fallback: brute-force probe common patterns
    if not found["phone"] and not found["aadhar"]:
        print("🔎 Falling back to brute-force probe...")
        for prefix, key in [("idx_phone", "phone"), ("idx_aadhar", "aadhar"),
                            ("phone", "phone"), ("aadhar", "aadhar"),
                            ("data", "other")]:
            for i in range(20):
                for fmt in [f"{prefix}.{i}.parquet", f"{prefix}_{i}.parquet",
                            f"{prefix}-{i}.parquet"]:
                    url = f"{HF_DATASET_URL}/{fmt}"
                    try:
                        # HEAD request to check existence
                        rr = await client.head(url, follow_redirects=True, timeout=15)
                        if rr.status_code == 200:
                            found[key].append(url)
                            print(f"✅ Found: {fmt}")
                    except Exception:
                        pass
    
    with _parquet_lock:
        _parquet_cache = found
    print(f"📦 Discovered: phone={len(found['phone'])}, aadhar={len(found['aadhar'])}, other={len(found['other'])}")
    return found

# ── Cache & Thread Pool ─────────────────────────────────────────────────────
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="search")
_df_cache: dict[str, pd.DataFrame] = {}
_df_cache_lock = threading.Lock()

async def download_parquet(url: str, client: httpx.AsyncClient) -> pd.DataFrame:
    """Download and cache parquet file."""
    with _df_cache_lock:
        if url in _df_cache:
            return _df_cache[url]
    
    try:
        response = await client.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        buffer = BytesIO(response.content)
        table = pq.read_table(buffer)
        df = table.to_pandas()
        with _df_cache_lock:
            _df_cache[url] = df
        print(f"✅ Loaded {url} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return pd.DataFrame()

async def search_in_parquet(field: str, value: str, limit: int = 10) -> list:
    """Search parquet files with auto-discovery."""
    results = []
    
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        files_map = await discover_parquet_files(client)
        
        # Pick relevant files
        if field == "aadharNumber":
            files = files_map["aadhar"] + files_map["other"]
        elif field in ("phoneNumber", "otherNumber"):
            files = files_map["phone"] + files_map["other"]
        else:
            files = files_map["phone"] + files_map["aadhar"] + files_map["other"]
        
        if not files:
            print("⚠️ No parquet files discovered!")
            return []
        
        # Search in parallel within this call
        async def search_one(url):
            df = await download_parquet(url, client)
            if df.empty or field not in df.columns:
                return []
            mask = df[field].astype(str).str.strip() == str(value).strip()
            return df[mask].to_dict("records")
        
        tasks = [search_one(u) for u in files]
        for coro in asyncio.as_completed(tasks):
            try:
                rows = await coro
                for r in rows:
                    results.append(r)
                    if len(results) >= limit:
                        return results
            except Exception as e:
                print(f"⚠️ Search error: {e}")
    
    return results

# ── Dedup & Connected Records ───────────────────────────────────────────────
def _connected_numbers(row: dict) -> list[dict]:
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        value = str(raw).strip()
        if not value or value in seen or value.lower() == "nan":
            continue
        seen.add(value)
        connected.append({"field": field, "value": value})
    return connected

def _cap_duplicates(rows: list[dict]) -> list[dict]:
    seen = {}
    out = []
    for r in rows:
        ph = str(r.get("phoneNumber", "")).strip()
        ad = str(r.get("aadharNumber", "")).strip()
        key = (ph, ad) if (ph or ad) else (str(r.get("name", "")), str(r.get("fathersName", "")))
        n = seen.get(key, 0)
        if n < DUPLICATE_CAP:
            seen[key] = n + 1
            record = dict(r)
            record["connected_numbers"] = _connected_numbers(record)
            out.append(record)
    return out

# ── Search Logic ────────────────────────────────────────────────────────────
def _run_async(coro):
    """Run async coroutine from sync code safely."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside a running loop; use a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as ex:
                return ex.submit(lambda: asyncio.run(coro)).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

def _unified_search_sync(q: str, limit: int = 10) -> dict:
    q = q.strip()
    is_num = q.isdigit() and len(q) >= 8
    
    if not is_num:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}
    
    all_rows = []
    searched = []
    
    # Try as phone
    if 8 <= len(q) <= 13:
        try:
            results = _run_async(search_in_parquet("phoneNumber", q, limit))
            if results:
                all_rows.extend(results)
                searched.append("phoneNumber")
        except Exception as e:
            print(f"Phone search error: {e}")
    
    # Try as aadhar
    if len(q) == 12:
        try:
            results = _run_async(search_in_parquet("aadharNumber", q, limit))
            if results:
                all_rows.extend(results)
                searched.append("aadharNumber")
        except Exception as e:
            print(f"Aadhar search error: {e}")
    
    # Try otherNumber as fallback
    if not all_rows:
        try:
            results = _run_async(search_in_parquet("otherNumber", q, limit))
            if results:
                all_rows.extend(results)
                searched.append("otherNumber")
        except Exception as e:
            print(f"otherNumber search error: {e}")
    
    all_rows = _cap_duplicates(all_rows)[:limit]
    
    return {
        "query": q,
        "searched_fields": searched,
        "count": len(all_rows),
        "results": all_rows,
    }

def _run_field_search_sync(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        return {"field": field, "value": value, "mode": mode, "count": 0, "results": [], "error": "Unknown field"}
    try:
        results = _run_async(search_in_parquet(field, value, limit))
        results = _cap_duplicates(results)[:limit]
        return {"field": field, "value": value, "mode": mode, "count": len(results), "results": results}
    except Exception as e:
        return {"field": field, "value": value, "mode": mode, "count": 0, "results": [], "error": str(e)}

# ── Pinger ──────────────────────────────────────────────────────────────────
async def pinger():
    port = os.getenv("PORT", "7860")
    url = f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(120)
            try:
                resp = await client.get(url)
                print(f"[Pinger] Status: {resp.status_code}")
            except Exception as e:
                print(f"[Pinger] Error: {e}")

# ── FastAPI Lifespan ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 ICMR Search API started!")
    # Pre-warm discovery
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            await discover_parquet_files(client)
    except Exception as e:
        print(f"Pre-warm failed: {e}")
    asyncio.create_task(pinger())
    yield
    print("👋 Shutting down...")

# ── FastAPI App ─────────────────────────────────────────────────────────────
fastapi_app = FastAPI(title="ICMR + HITEK Search API", lifespan=lifespan)

class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10

@fastapi_app.get("/")
def root():
    return {
        "app": "ICMR + HITEK Search API",
        "dataset": "rehuuuu/icrm-hitek-fulldb",
        "columns": SEARCH_FIELDS,
        "docs": "/docs",
        "developer": "@kzr0x | channel @api_wallah",
    }

@fastapi_app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset": "rehuuuu/icrm-hitek-fulldb",
        "discovered_files": {k: len(v) for k, v in _parquet_cache.items()},
        "cached_dfs": len(_df_cache),
    }

@fastapi_app.get("/debug/files")
async def debug_files():
    """Debug endpoint to check discovered parquet files."""
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        files = await discover_parquet_files(client)
    return files

@fastapi_app.get("/search")
async def search(
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    aadhar: str | None = Query(None),
    field: str | None = Query(None),
    mode: str = Query("exact"),
    limit: int = Query(10, ge=1, le=100),
    pretty: bool = Query(True),
):
    if aadhar:
        q_val = aadhar.strip(); field = "aadharNumber"
    elif mobile:
        q_val = mobile.strip(); field = "phoneNumber"
    elif q:
        q_val = q.strip()
    else:
        raise HTTPException(422, "Provide q, mobile, or aadhar")
    
    if not q_val:
        raise HTTPException(422, "Query cannot be empty")
    
    loop = asyncio.get_running_loop()
    if field:
        data = await loop.run_in_executor(pool, _run_field_search_sync, field, q_val, mode, limit)
    else:
        data = await loop.run_in_executor(pool, _unified_search_sync, q_val, limit)
    
    result = {"success": bool(data.get("count", 0) > 0), **data, "number": q_val}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False, default=str)
    return Response(content=content, media_type="application/json")

@fastapi_app.get("/search/phone/{number}")
async def search_phone(number: str, limit: int = Query(10, ge=1, le=100), pretty: bool = Query(True)):
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, _run_field_search_sync, "phoneNumber", number, "exact", limit)
    result = {"success": bool(data.get("count", 0) > 0), **data, "number": number}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False, default=str)
    return Response(content=content, media_type="application/json")

@fastapi_app.get("/search/aadhar/{number}")
async def search_aadhar(number: str, limit: int = Query(10, ge=1, le=100), pretty: bool = Query(True)):
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, _run_field_search_sync, "aadharNumber", number, "exact", limit)
    result = {"success": bool(data.get("count", 0) > 0), **data, "number": number}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False, default=str)
    return Response(content=content, media_type="application/json")

@fastapi_app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 20:
        raise HTTPException(400, "max 20 queries per batch")
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(pool, _run_field_search_sync,
                             item.get("field", "phoneNumber"),
                             item.get("value", ""),
                             item.get("mode", "exact"),
                             int(item.get("limit", req.limit)))
        for item in req.queries
    ]
    results = await asyncio.gather(*tasks)
    return {"searches": len(req.queries), "results": list(results)}

# ── Gradio UI ───────────────────────────────────────────────────────────────
def format_result(row: dict) -> str:
    lines = []
    for field in SEARCH_FIELDS:
        val = row.get(field, "")
        if val and str(val) != "nan":
            lines.append(f"**{field}:** {val}")
    cn = row.get("connected_numbers", [])
    if cn:
        nums = ", ".join(f"{c['field']}={c['value']}" for c in cn)
        lines.append(f"**connected:** {nums}")
    return "\n\n".join(lines)

def search_ui(query: str, limit: int) -> str:
    if not query or not query.strip():
        return "⚠️ Kuch toh search karo — phone ya aadhar number daalo."
    q = query.strip()
    data = _unified_search_sync(q, int(limit))
    count = data.get("count", 0)
    results = data.get("results", [])
    searched = ", ".join(data.get("searched_fields", []))
    if not results:
        return f"🔍 **Query:** `{q}`\n**Searched:** {searched or 'none'}\n\n❌ **No data found** for this number.\n\n_Tip: check /debug/files to verify parquet files are being discovered._"
    header = f"🔍 **Query:** `{q}`  |  **Found:** {count} results  |  **Searched:** {searched}\n\n---\n\n"
    parts = [f"### Result {i}\n{format_result(row)}" for i, row in enumerate(results, 1)]
    return header + "\n\n---\n\n".join(parts)

def build_ui():
    with gr.Blocks(title="ICMR Search API", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 ICMR + HITEK Search API")
        gr.Markdown("Search **rehuuuu/icrm-hitek-fulldb** — phone, Aadhaar & more")
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(label="Search Query", placeholder="Phone number ya Aadhaar daalo...", lines=1)
            with gr.Column(scale=1):
                limit_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Max Results")
        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
        output = gr.Markdown(label="Results")
        search_btn.click(fn=search_ui, inputs=[query_input, limit_slider], outputs=output)
        query_input.submit(fn=search_ui, inputs=[query_input, limit_slider], outputs=output)
        gr.Markdown("---")
        with gr.Accordion("📡 API Info", open=False):
            gr.Markdown("""
**Endpoints:**
- `GET /search?q=<number>` — Auto-detect search
- `GET /search/phone/<number>` — Phone search
- `GET /search/aadhar/<number>` — Aadhar search
- `GET /health` — Health check
- `GET /debug/files` — See discovered parquet files
- `GET /docs` — Swagger UI
            """)
        gr.Markdown("---\n<div style='text-align:center;color:#888;'>👨‍💻 **Developer:** @kzr0x | 📢 **Channel:** @api_wallah</div>")
    return demo

demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
