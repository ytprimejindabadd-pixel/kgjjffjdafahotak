import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import gradio as gr
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
HF_DATASET = os.environ.get(
    "ICMR_HF_DATASET",
    "rehuuuu/icrm-hitek-full-db-mixed"
)
HF_API_BASE = "https://datasets-server.huggingface.co"
PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "5"))
TIMEOUT = int(os.environ.get("ICMR_TIMEOUT", "30"))
MAX_ROWS_PER_REQUEST = 100  # HF API max limit
DUPLICATE_CAP = 2

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

# ── HTTP Client ─────────────────────────────────────────────────────────────
client = httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True)

# ── HuggingFace API Functions ──────────────────────────────────────────────
async def fetch_dataset_rows(offset: int = 0, length: int = 100) -> dict:
    """Fetch rows from HuggingFace datasets-server API"""
    url = f"{HF_API_BASE}/rows"
    params = {
        "dataset": HF_DATASET,
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": min(length, MAX_ROWS_PER_REQUEST)
    }
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


async def search_in_dataset(query: str, limit: int = 10) -> dict:
    """Search for a query in the HF dataset"""
    # Normalize query
    query = query.strip()
    if not query:
        return {"query": query, "searched_fields": [], "count": 0, "results": []}
    
    is_number = query.isdigit() and len(query) >= 8
    searched_fields = []
    results = []
    
    if not is_number:
        return {"query": query, "searched_fields": [], "count": 0, "results": []}
    
    # For phone number (10 digits)
    if len(query) == 10:
        searched_fields.append("phoneNumber")
        # Search by phone
        phone_results = await search_by_field("phoneNumber", query, limit)
        results.extend(phone_results)
    
    # For Aadhar number (12 digits)
    if len(query) == 12:
        searched_fields.append("aadharNumber")
        # Search by Aadhar
        aadhar_results = await search_by_field("aadharNumber", query, limit)
        results.extend(aadhar_results)
    
    # If no results and it could be other number
    if not results and is_number:
        searched_fields.append("otherNumber")
        other_results = await search_by_field("otherNumber", query, limit)
        results.extend(other_results)
    
    # Remove duplicates
    unique_results = []
    seen = set()
    for row in results:
        key = (row.get("phoneNumber", ""), row.get("aadharNumber", ""))
        if key not in seen:
            seen.add(key)
            row["connected_numbers"] = get_connected_numbers(row)
            unique_results.append(row)
        if len(unique_results) >= limit:
            break
    
    return {
        "query": query,
        "searched_fields": searched_fields,
        "count": len(unique_results),
        "results": unique_results
    }


async def search_by_field(field: str, value: str, limit: int = 10) -> list:
    """Search by specific field using HF API with pagination"""
    results = []
    offset = 0
    max_pages = 10  # Search through first 1000 rows max
    
    for page in range(max_pages):
        data = await fetch_dataset_rows(offset, MAX_ROWS_PER_REQUEST)
        
        if "error" in data:
            break
            
        rows = data.get("rows", [])
        if not rows:
            break
        
        # Filter rows
        for row_data in rows:
            row = row_data.get("row", {})
            if str(row.get(field, "")).strip() == value:
                results.append(row)
                if len(results) >= limit:
                    return results
        
        offset += MAX_ROWS_PER_REQUEST
    
    return results


def get_connected_numbers(row: dict) -> list:
    """Extract connected numbers from row"""
    connected = []
    seen = set()
    for field in NUMBER_FIELDS:
        value = str(row.get(field, "")).strip()
        if value and value not in seen:
            seen.add(value)
            connected.append({"field": field, "value": value})
    return connected


# ── Sync Search Function (for Gradio) ──────────────────────────────────────
def unified_search_sync(query: str, limit: int = 10) -> dict:
    """Synchronous wrapper for search"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(search_in_dataset(query, limit))
        loop.close()
        return result
    except Exception as e:
        return {"query": query, "searched_fields": [], "count": 0, "results": [], "error": str(e)}


# ── FastAPI Lifespan ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting ICMR Search API with HuggingFace Dataset...")
    yield
    # Shutdown
    await client.aclose()
    print("👋 Shutting down...")


# ── FastAPI App ─────────────────────────────────────────────────────────────
fastapi_app = FastAPI(title="ICMR + HITEK Search API (HF Dataset)", lifespan=lifespan)


class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10


@fastapi_app.get("/")
def root():
    return {
        "app": "ICMR + HITEK Search API",
        "dataset": HF_DATASET,
        "source": "HuggingFace Datasets Server",
        "columns": SEARCH_FIELDS,
        "docs": "/docs",
        "developer": "@kzr0x | channel @api_wallah",
    }


@fastapi_app.get("/health")
async def health():
    """Health check with dataset info"""
    try:
        # Check dataset availability
        splits_url = f"{HF_API_BASE}/splits"
        response = await client.get(splits_url, params={"dataset": HF_DATASET})
        dataset_status = "ok" if response.status_code == 200 else "error"
        splits = response.json().get("splits", []) if response.status_code == 200 else []
    except:
        dataset_status = "unavailable"
        splits = []
    
    return {
        "status": "ok",
        "dataset": HF_DATASET,
        "dataset_status": dataset_status,
        "splits": splits,
        "index_source": "huggingface_datasets_server"
    }


@fastapi_app.get("/search")
async def search(
    q: str | None = Query(None, description="Search query (phone/aadhar number)"),
    mobile: str | None = Query(None, description="Mobile number to search"),
    aadhar: str | None = Query(None, description="Aadhar number to search"),
    field: str | None = Query(None, description="Specific field to search"),
    mode: str = Query("exact", description="Search mode (exact/contains)"),
    limit: int = Query(10, ge=1, le=100, description="Results limit"),
    pretty: bool = Query(True, description="Pretty print JSON"),
):
    """Search endpoint"""
    # Determine query
    if aadhar:
        q_val = aadhar.strip()
        field = "aadharNumber"
    elif mobile:
        q_val = mobile.strip()
        field = "phoneNumber"
    elif q:
        q_val = q.strip()
    else:
        raise HTTPException(422, "Provide q, mobile, or aadhar parameter")
    
    if not q_val:
        raise HTTPException(422, "Query cannot be empty")
    
    # Search
    if field:
        results = await search_by_field(field, q_val, limit)
        data = {
            "field": field,
            "value": q_val,
            "mode": mode,
            "count": len(results),
            "results": results
        }
    else:
        data = await search_in_dataset(q_val, limit)
    
    result = {
        "success": bool(data.get("count", 0) > 0),
        **data,
        "number": q_val,
        "total": data.get("count", 0)
    }
    
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


@fastapi_app.get("/search/phone/{number}")
async def search_phone(
    number: str,
    limit: int = Query(10, ge=1, le=100),
    pretty: bool = Query(True)
):
    """Search by phone number"""
    results = await search_by_field("phoneNumber", number.strip(), limit)
    result = {
        "success": len(results) > 0,
        "field": "phoneNumber",
        "value": number,
        "count": len(results),
        "results": results,
        "total": len(results)
    }
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


@fastapi_app.get("/search/aadhar/{number}")
async def search_aadhar(
    number: str,
    limit: int = Query(10, ge=1, le=100),
    pretty: bool = Query(True)
):
    """Search by Aadhar number"""
    results = await search_by_field("aadharNumber", number.strip(), limit)
    result = {
        "success": len(results) > 0,
        "field": "aadharNumber",
        "value": number,
        "count": len(results),
        "results": results,
        "total": len(results)
    }
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


@fastapi_app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    """Batch search endpoint"""
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 20:
        raise HTTPException(400, "max 20 queries per batch")
    
    tasks = []
    for item in req.queries:
        field = item.get("field", "phoneNumber")
        value = item.get("value", "")
        limit = int(item.get("limit", req.limit))
        tasks.append(search_by_field(field, value, limit))
    
    results = await asyncio.gather(*tasks)
    
    return {
        "searches": len(req.queries),
        "results": [
            {
                "field": req.queries[i].get("field", "phoneNumber"),
                "value": req.queries[i].get("value", ""),
                "count": len(results[i]),
                "results": results[i]
            }
            for i in range(len(req.queries))
        ]
    }


@fastapi_app.get("/dataset/info")
async def dataset_info():
    """Get dataset information"""
    try:
        # Get splits
        splits_response = await client.get(
            f"{HF_API_BASE}/splits",
            params={"dataset": HF_DATASET}
        )
        splits = splits_response.json() if splits_response.status_code == 200 else {}
        
        # Get first rows sample
        rows_response = await client.get(
            f"{HF_API_BASE}/rows",
            params={
                "dataset": HF_DATASET,
                "config": "default",
                "split": "train",
                "offset": 0,
                "length": 5
            }
        )
        sample_rows = rows_response.json() if rows_response.status_code == 200 else {}
        
        return {
            "dataset": HF_DATASET,
            "splits": splits,
            "sample": sample_rows,
            "api_base": HF_API_BASE
        }
    except Exception as e:
        raise HTTPException(500, f"Error fetching dataset info: {str(e)}")


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
    data = unified_search_sync(q, int(limit))
    
    if "error" in data:
        return f"❌ Error: {data['error']}"
    
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
    with gr.Blocks(title="ICMR Search API") as demo:
        gr.Markdown("# 🔍 ICMR + HITEK Search API", elem_classes="main-title")
        gr.Markdown(f"Search **{HF_DATASET}** — phone, Aadhaar & more", elem_classes="subtitle")
        
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Phone number (10 digits) ya Aadhaar (12 digits) daalo...",
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
            gr.Markdown(f"""
**Endpoints** (via FastAPI):
- `GET /search?q=<number>` — Phone/Aadhaar search
- `GET /search/phone/{{number}}` — Phone-specific search
- `GET /search/aadhar/{{number}}` — Aadhar-specific search
- `GET /health` — Health check
- `GET /docs` — Swagger UI

**Dataset:** {HF_DATASET}
**Source:** HuggingFace Datasets Server API
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
    print(f"📊 Dataset: {HF_DATASET}")
    print(f"🔗 API Docs: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)
