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

# Parquet file URLs (adjust based on actual file structure)
PARQUET_FILES = {
    "phone": [f"{HF_DATASET_URL}/idx_phone.{i}.parquet" for i in range(7)],
    "aadhar": [f"{HF_DATASET_URL}/idx_aadhar.{i}.parquet" for i in range(7)],
}

# ── Cache & Thread Pool ─────────────────────────────────────────────────────
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="search")
_cache = {}
_cache_lock = threading.Lock()

def _get_cached_data():
    """Get or initialize cached data for current thread"""
    if not hasattr(_thread_local, "data"):
        _thread_local.data = {}
    return _thread_local.data

async def download_parquet(url: str) -> pd.DataFrame:
    """Download and read parquet file"""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Read parquet from bytes
            buffer = BytesIO(response.content)
            table = pq.read_table(buffer)
            return table.to_pandas()
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return pd.DataFrame()

async def search_in_parquet(field: str, value: str, limit: int = 10) -> list:
    """Search in parquet files"""
    results = []
    files = PARQUET_FILES.get("phone", []) if field in ["phoneNumber", "otherNumber"] else PARQUET_FILES.get("aadhar", [])
    
    # If searching phone, also check aadhar files if needed
    if field == "phoneNumber":
        files = PARQUET_FILES.get("phone", [])
    elif field == "aadharNumber":
        files = PARQUET_FILES.get("aadhar", [])
    
    for url in files[:3]:  # Limit to first 3 files for speed
        try:
            df = await download_parquet(url)
            if df.empty:
                continue
                
            # Search in dataframe
            if field in df.columns:
                mask = df[field].astype(str).str.strip() == value
                matches = df[mask]
                
                for _, row in matches.iterrows():
                    result = row.to_dict()
                    results.append(result)
                    if len(results) >= limit:
                        return results
        except Exception as e:
            print(f"⚠️ Error searching {url}: {e}")
            continue
    
    return results

# ── Dedup & Connected Records ───────────────────────────────────────────────
def _connected_numbers(row: dict) -> list[dict]:
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None or pd.isna(raw):
            continue
        value = str(raw).strip()
        if not value or value in seen:
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
        key = (ph, ad) if ph or ad else (str(r.get("name", "")), str(r.get("fathersName", "")))
        
        n = seen.get(key, 0)
        if n < DUPLICATE_CAP:
            seen[key] = n + 1
            record = dict(r)
            record["connected_numbers"] = _connected_numbers(record)
            out.append(record)
    return out

# ── Search Logic ────────────────────────────────────────────────────────────
def _unified_search_sync(q: str, limit: int = 10) -> dict:
    """Synchronous search function"""
    q = q.strip()
    is_num = q.isdigit() and len(q) >= 8
    
    if not is_num:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}
    
    all_rows = []
    searched = []
    
    # Search in phone files
    if len(q) == 10 or len(q) == 11:  # Phone number
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(search_in_parquet("phoneNumber", q, limit))
            loop.close()
            
            if results:
                all_rows.extend(results)
                searched.append("phoneNumber")
        except Exception as e:
            print(f"Phone search error: {e}")
    
    # Search in aadhar files
    if len(q) == 12:  # Aadhar number
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(search_in_parquet("aadharNumber", q, limit))
            loop.close()
            
            if results:
                all_rows.extend(results)
                searched.append("aadharNumber")
        except Exception as e:
            print(f"Aadhar search error: {e}")
    
    all_rows = _cap_duplicates(all_rows)[:limit]
    
    return {
        "query": q,
        "searched_fields": searched,
        "count": len(all_rows),
        "results": all_rows
    }

def _run_field_search_sync(field: str, value: str, mode: str, limit: int) -> dict:
    """Synchronous field search"""
    if field not in SEARCH_FIELDS:
        return {"field": field, "value": value, "mode": mode, "count": 0, "results": [], "error": "Unknown field"}
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(search_in_parquet(field, value, limit))
        loop.close()
        
        results = _cap_duplicates(results)[:limit]
        return {"field": field, "value": value, "mode": mode, "count": len(results), "results": results}
    except Exception as e:
        return {"field": field, "value": value, "mode": mode, "count": 0, "results": [], "error": str(e)}

# ── Pinger ──────────────────────────────────────────────────────────────────
async def pinger():
    """Ping the /health endpoint every 2 minutes"""
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
        "searchable_fields": ["phoneNumber", "aadharNumber"]
    }

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
        q_val = aadhar.strip()
        field = "aadharNumber"
    elif mobile:
        q_val = mobile.strip()
        field = "phoneNumber"
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
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")

@fastapi_app.get("/search/phone/{number}")
async def search_phone(number: str, limit: int = Query(10, ge=1, le=100), pretty: bool = Query(True)):
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, _run_field_search_sync, "phoneNumber", number, "exact", limit)
    result = {"success": bool(data.get("count", 0) > 0), **data, "number": number}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")

@fastapi_app.get("/search/aadhar/{number}")
async def search_aadhar(number: str, limit: int = Query(10, ge=1, le=100), pretty: bool = Query(True)):
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, _run_field_search_sync, "aadharNumber", number, "exact", limit)
    result = {"success": bool(data.get("count", 0) > 0), **data, "number": number}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
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
        return f"🔍 **Query:** `{q}`\n**Searched:** {searched}\n\n❌ **No data found** for this number."
    
    header = f"🔍 **Query:** `{q}`  |  **Found:** {count} results  |  **Searched:** {searched}\n\n---\n\n"
    parts = []
    for i, row in enumerate(results, 1):
        parts.append(f"### Result {i}\n{format_result(row)}")
    
    return header + "\n\n---\n\n".join(parts)

def build_ui():
    with gr.Blocks(title="ICMR Search API", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 ICMR + HITEK Search API")
        gr.Markdown("Search **rehuuuu/icrm-hitek-fulldb** — phone, Aadhaar & more")
        
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Phone number ya Aadhaar daalo...",
                    lines=1,
                )
            with gr.Column(scale=1):
                limit_slider = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1,
                    label="Max Results",
                )
        
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
- `GET /docs` — Swagger UI

**Dataset:** rehuuuu/icrm-hitek-fulldb
            """)
        
        gr.Markdown("---\n<div style='text-align:center;color:#888;'>👨‍💻 **Developer:** @kzr0x | 📢 **Channel:** @api_wallah</div>")
    
    return demo

# ── Mount Gradio on FastAPI ─────────────────────────────────────────────────
demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# ── Main Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Starting server on port {port}")
    print(f"📊 Dataset: rehuuuu/icrm-hitek-fulldb")
    print(f"🔗 API Docs: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)
