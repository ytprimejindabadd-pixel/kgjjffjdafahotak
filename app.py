import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor

import duckdb
import gradio as gr
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
HF_INDEX_BASE = os.environ.get(
    "ICMR_HF_INDEX_BASE",
    "https://huggingface.co/datasets/rehuuuu/icrm-hitek-fulldb/resolve/main",
).rstrip("/")
INDEX_SOURCE = os.environ.get("ICMR_INDEX_SOURCE", "remote").lower()
PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "2"))
THREADS_PER_CONN = int(os.environ.get("ICMR_THREADS_PER_CONN", "2"))
DUPLICATE_CAP = 2

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

IDX_PHONE = "idx_phone"
IDX_AADHAR = "idx_aadhar"

# Try to detect available parquet files
REMOTE_INDEXES = {
    "phone": [f"{HF_INDEX_BASE}/idx_phone.{i}.parquet" for i in range(7)],
    "aadhar": [f"{HF_INDEX_BASE}/idx_aadhar.{i}.parquet" for i in range(7)],
}

# ── DuckDB Connection Pool ──────────────────────────────────────────────────
_conns: list[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")

def _idx_ready(kind: str) -> bool:
    return kind in REMOTE_INDEXES

def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    # Vercel fix: set home & extension dir to /tmp
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    
    # Try to create views for available indexes
    for kind, urls in REMOTE_INDEXES.items():
        view = f"people_{kind}"
        try:
            lst = ", ".join(f"'{u}'" for u in urls)
            con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet([{lst}])")
        except Exception as e:
            print(f"⚠️ Failed to create {view} view: {e}")
    
    con.execute(f"SET threads = {THREADS_PER_CONN}")
    return con

def _thread_id() -> int:
    tid = getattr(_thread_local, "id", None)
    if tid is None:
        with _conns_lock:
            tid = len(_conns)
            _thread_local.id = tid
    return tid

def _get_conn() -> duckdb.DuckDBPyConnection:
    ident = _thread_id()
    with _conns_lock:
        while len(_conns) <= ident:
            _conns.append(_new_conn())
    return _conns[ident]

# ── Dedup & Connected Records ───────────────────────────────────────────────
def _person_key(row: dict) -> tuple:
    ph = (row.get("phoneNumber") or "").strip()
    ad = (row.get("aadharNumber") or "").strip()
    if ph or ad:
        return (ph, ad)
    return (row.get("name") or "").strip(), (row.get("fathersName") or "").strip()

def _connected_numbers(row: dict) -> list[dict]:
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None:
            continue
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        connected.append({"field": field, "value": value})
    return connected

def _cap_duplicates(rows: list[dict]) -> list[dict]:
    seen: dict[tuple, int] = {}
    out = []
    for r in rows:
        k = _person_key(r)
        n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1
            record = dict(r)
            record["connected_numbers"] = _connected_numbers(record)
            out.append(record)
    return out

# ── Search Logic ────────────────────────────────────────────────────────────
def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    v = value.replace("'", "''")

    if mode == "exact":
        if field == "phoneNumber" and _idx_ready("phone"):
            view = "people_phone"
        elif field == "aadharNumber" and _idx_ready("aadhar"):
            view = "people_aadhar"
        elif field == "otherNumber":
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        else:
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        sql = f"SELECT * FROM {view} WHERE {field} = '{v}' LIMIT {limit * DUPLICATE_CAP + 20}"
    elif mode == "contains":
        if field == "name":
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        v2 = v.replace("%", r"\%").replace("_", r"\_")
        sql = f"SELECT * FROM people_phone WHERE {field} ILIKE '%{v2}%' ESCAPE '\\' LIMIT {limit * DUPLICATE_CAP + 20}"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    try:
        con = _get_conn()
        rows = con.execute(sql).fetchall()
        cols = [d[0] for d in con.description]
        results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
        return {"field": field, "value": value, "mode": mode, "count": len(results), "results": results}
    except Exception as e:
        return {"field": field, "value": value, "mode": mode, "count": 0, "results": [], "error": str(e)}

def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    is_num = q.isdigit() and len(q) >= 8

    if is_num:
        all_rows = []
        searched = []
        
        # Phone index first (fast)
        if _idx_ready("phone"):
            r = _run_field_search("phoneNumber", q, "exact", limit)
            if not r.get("error"):
                all_rows.extend(r["results"])
                searched.append("phoneNumber")
        
        # Aadhar index second
        if not all_rows and _idx_ready("aadhar"):
            r = _run_field_search("aadharNumber", q, "exact", limit)
            if not r.get("error"):
                all_rows.extend(r["results"])
                searched.append("aadharNumber")
        
        all_rows = _cap_duplicates(all_rows)[:limit]
        return {
            "query": q, "searched_fields": searched,
            "count": len(all_rows), "results": all_rows,
        }
    else:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}

# ── Pinger (keeps app alive) ──────────────────────────────────────────────
async def pinger():
    """Ping the /health endpoint every 2 minutes to prevent idle shutdown."""
    port = os.getenv("PORT", "7860")
    url = f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(120)
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    print(f"[Pinger] OK")
                else:
                    print(f"[Pinger] Unexpected status: {resp.status_code}")
            except Exception as e:
                print(f"[Pinger] Error: {e}")

# ── FastAPI Lifespan ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    asyncio.create_task(pinger())
    print("🚀 ICMR Search API started!")
    yield
    # Shutdown
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
        "indexes": {"phone": _idx_ready("phone"), "aadhar": _idx_ready("aadhar")},
        "index_source": INDEX_SOURCE,
        "columns": SEARCH_FIELDS,
        "docs": "/docs",
        "developer": "@kzr0x | channel @api_wallah",
    }

@fastapi_app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset": "rehuuuu/icrm-hitek-fulldb",
        "indexes": {"phone": _idx_ready("phone"), "aadhar": _idx_ready("aadhar")},
        "index_source": INDEX_SOURCE
    }

@fastapi_app.get("/search")
async def search(
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    aadhar: str | None = Query(None),
    field: str | None = Query(None),
    mode: str = Query("exact"),
    limit: int = Query(10, ge=1, le=1000),
    pretty: bool = Query(True),
):
    # Priority: aadhar > mobile > q
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
        data = await loop.run_in_executor(pool, _run_field_search, field, q_val, mode, limit)
    else:
        data = await loop.run_in_executor(pool, _unified_search, q_val, limit)
    
    result = {"success": bool(data.get("count", 0) > 0), **data, "number": q_val,
              "total": data.get("count", 0)}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")

@fastapi_app.get("/search/phone/{number}")
async def search_phone(
    number: str,
    limit: int = Query(10, ge=1, le=1000),
    pretty: bool = Query(True)
):
    """Search by phone number"""
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, _run_field_search, "phoneNumber", number, "exact", limit)
    result = {"success": bool(data.get("count", 0) > 0), **data, "number": number}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")

@fastapi_app.get("/search/aadhar/{number}")
async def search_aadhar(
    number: str,
    limit: int = Query(10, ge=1, le=1000),
    pretty: bool = Query(True)
):
    """Search by Aadhar number"""
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, _run_field_search, "aadharNumber", number, "exact", limit)
    result = {"success": bool(data.get("count", 0) > 0), **data, "number": number}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")

@fastapi_app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 50:
        raise HTTPException(400, "max 50 queries per batch")
    
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(pool, _run_field_search,
                             item.get("field", "phoneNumber"),
                             item.get("value", ""),
                             item.get("mode", "exact"),
                             int(item.get("limit", req.limit)))
        for item in req.queries
    ]
    results = await asyncio.gather(*tasks)
    return Response(content=json.dumps({"searches": len(req.queries), "results": list(results)},
                                       indent=2, ensure_ascii=False),
                    media_type="application/json")

# ── Gradio UI ───────────────────────────────────────────────────────────────
def format_result(row: dict) -> str:
    """Format a single result record as readable text."""
    lines = []
    for field in SEARCH_FIELDS:
        val = row.get(field, "")
        if val:
            lines.append(f"**{field}:** {val}")
    
    cn = row.get("connected_numbers", [])
    if cn:
        nums = ", ".join(f"{c['field']}={c['value']}" for c in cn)
        lines.append(f"**connected:** {nums}")
    
    return "\n\n".join(lines)

def search_ui(query: str, limit: int) -> str:
    """Main Gradio search function."""
    if not query or not query.strip():
        return "⚠️ Kuch toh search karo — phone ya aadhar number daalo."
    
    q = query.strip()
    try:
        data = _unified_search(q, int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"
    
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
    with gr.Blocks(
        title="ICMR Search API",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 0; }
        .subtitle { text-align: center; color: #666; margin-top: 0; }
        .footer { text-align: center; color: #888; margin-top: 20px; }
        """
    ) as demo:
        gr.Markdown("# 🔍 ICMR + HITEK Search API", elem_classes="main-title")
        gr.Markdown("Search **rehuuuu/icrm-hitek-fulldb** — phone, Aadhaar & more", elem_classes="subtitle")
        
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Phone number ya Aadhaar daalo...",
                    lines=1,
                )
            with gr.Column(scale=1):
                limit_slider = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="Max Results",
                )
        
        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
        output = gr.Markdown(label="Results")
        
        search_btn.click(
            fn=search_ui,
            inputs=[query_input, limit_slider],
            outputs=output,
        )
        query_input.submit(
            fn=search_ui,
            inputs=[query_input, limit_slider],
            outputs=output,
        )
        
        gr.Markdown("---")
        with gr.Accordion("📡 API Info", open=False):
            gr.Markdown("""
**Endpoints** (via FastAPI):
- `GET /search?q=<number>` — Phone/Aadhaar search
- `GET /search/phone/<number>` — Phone-specific search
- `GET /search/aadhar/<number>` — Aadhar-specific search
- `GET /health` — Health check
- `GET /docs` — Swagger UI

**Dataset:** rehuuuu/icrm-hitek-fulldb
            """)
        
        gr.Markdown(
            "---\n"
            "<div class='footer'>"
            "👨‍💻 **Developer:** @kzr0x  |  📢 **Channel:** @api_wallah"
            "</div>",
            elem_classes="footer"
        )
    
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
