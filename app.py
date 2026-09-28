from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import duckdb
import os
import re
import logging
import urllib.request
from contextlib import asynccontextmanager

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

# ---------- Config ----------
PARQUET_URL = "https://huggingface.co/datasets/tfqdeadlo/Inddatainonefile/resolve/main/users_data.parquet"
LOCAL_PARQUET = "/tmp/users_data.parquet"

SEARCH_COLUMNS = {
    "mobile": "mobile",
    "alt": "alt",
    "id": "id",
    "email": "email",
}
VALID_TYPES = list(SEARCH_COLUMNS.keys())

con = None

# ---------- Lifespan (startup / shutdown) ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global con
    logger.info("Starting app...")

    # Download parquet to local disk (fast queries)
    src = PARQUET_URL
    try:
        if not os.path.exists(LOCAL_PARQUET):
            logger.info("Downloading parquet from HuggingFace...")
            urllib.request.urlretrieve(PARQUET_URL, LOCAL_PARQUET)
            logger.info(f"Downloaded to {LOCAL_PARQUET}")
        else:
            logger.info("Using cached parquet")
        src = LOCAL_PARQUET
    except Exception as e:
        logger.warning(f"Download failed, streaming from remote URL: {e}")

    # DuckDB connection
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"CREATE OR REPLACE VIEW users AS SELECT * FROM read_parquet('{src}')")
    logger.info("DB ready ✅")

    yield

    con.close()
    logger.info("Shutdown complete")


# ---------- App ----------
app = FastAPI(lifespan=lifespan, title="User Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Routes ----------
@app.get("/")
def home():
    return {
        "status": "online",
        "message": "User Search API",
        "endpoints": {
            "search": "/search?type=mobile&q=9876543210&limit=100",
            "stats": "/stats",
            "health": "/health",
        },
        "types": VALID_TYPES,
    }


@app.get("/search")
def search(
    q: str = Query(..., description="Search value"),
    type: str = Query("mobile", description=f"Search type: {VALID_TYPES}"),
    limit: int = Query(100, description="Max results (1-500)"),
):
    q_clean = str(q).strip()
    type_clean = str(type).strip().lower()
    limit_clean = min(max(limit, 1), 500)

    if type_clean not in VALID_TYPES:
        return {"success": False, "message": f"Invalid type. Valid: {VALID_TYPES}"}

    # Validation rules
    rules = {
        "mobile": (r"^\d{10}$", "Mobile number must be 10 digits"),
        "alt":    (r"^\d{10}$", "Alternate number must be 10 digits"),
        "id":     (r"^\d{12}$", "Aadhar ID must be 12 digits"),
        "email":  (r"^[^@]+@[^@]+\.[^@]+$", "Invalid email format"),
    }
    pattern, msg = rules[type_clean]
    if not re.match(pattern, q_clean):
        return {"success": False, "message": msg}

    try:
        col = SEARCH_COLUMNS[type_clean]
        sql = f'''
            SELECT mobile, name, fname AS father_name, address, circle,
                   alt AS alternate, id AS aadhar, email
            FROM users
            WHERE "{col}" = ?
            LIMIT ?
        '''
        rows_raw = con.execute(sql, [q_clean, limit_clean]).fetchall()

        if not rows_raw:
            return {"success": False, "message": f"{type_clean} '{q_clean}' not found"}

        cols = ["mobile", "name", "father_name", "address", "circle",
                "alternate", "aadhar", "email"]
        rows = [dict(zip(cols, r)) for r in rows_raw]

        return {
            "success": True,
            "number": q_clean,
            "total": len(rows),
            "results": rows,
        }

    except Exception as e:
        logger.exception("Search failed")
        return {"success": False, "message": f"Internal error: {e}"}


@app.get("/stats")
def stats():
    try:
        count = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        return {"status": "success", "total_records": count}
    except Exception as e:
        logger.exception("Stats failed")
        return {"status": "error", "message": str(e)}


@app.get("/health")
def health():
    try:
        con.execute("SELECT 1").fetchone()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# ---------- Local run / Render run ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
