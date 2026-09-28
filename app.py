from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import duckdb
import os
import re
import logging
import urllib.request
from contextlib import asynccontextmanager

logger = logging.getLogger("uvicorn.error")

PARQUET_URL = "https://huggingface.co/datasets/tfqdeadlo/Inddatainonefile/resolve/main/users_data.parquet"
LOCAL_PARQUET = "/tmp/users_data.parquet"

SEARCH_COLUMNS = {"mobile": "mobile", "alt": "alt", "id": "id", "email": "email"}
VALID_TYPES = list(SEARCH_COLUMNS.keys())

con = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global con
    if not os.path.exists(LOCAL_PARQUET):
        logger.info("Downloading parquet...")
        urllib.request.urlretrieve(PARQUET_URL, LOCAL_PARQUET)
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"CREATE OR REPLACE VIEW users AS SELECT * FROM read_parquet('{LOCAL_PARQUET}')")
    logger.info("DB ready")
    yield
    con.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "online", "types": VALID_TYPES}

@app.get("/search")
def search(
    q: str = Query(...),
    type: str = Query("mobile"),
    limit: int = Query(100),
):
    q_clean = q.strip()
    type_clean = type.strip().lower()
    limit_clean = min(max(limit, 1), 500)

    if type_clean not in VALID_TYPES:
        return {"success": False, "message": f"Invalid type. Valid: {VALID_TYPES}"}

    rules = {
        "mobile": (r'^\d{10}$', "Mobile must be 10 digits"),
        "alt":    (r'^\d{10}$', "Alternate must be 10 digits"),
        "id":     (r'^\d{12}$', "Aadhar must be 12 digits"),
        "email":  (r'^[^@]+@[^@]+\.[^@]+$', "Invalid email"),
    }
    pattern, msg = rules[type_clean]
    if not re.match(pattern, q_clean):
        return {"success": False, "message": msg}

    try:
        col = SEARCH_COLUMNS[type_clean]
        sql = f'''
            SELECT mobile, name, fname AS father_name, address, circle,
                   alt AS alternate, id AS aadhar, email
            FROM users WHERE "{col}" = ? LIMIT ?
        '''
        rows_raw = con.execute(sql, [q_clean, limit_clean]).fetchall()
        if not rows_raw:
            return {"success": False, "message": f"{type_clean} '{q_clean}' not found"}

        cols = ['mobile','name','father_name','address','circle','alternate','aadhar','email']
        rows = [dict(zip(cols, r)) for r in rows_raw]
        return {"success": True, "number": q_clean, "total": len(rows), "results": rows}
    except Exception:
        logger.exception("search failed")
        return {"success": False, "message": "Internal error"}

@app.get("/stats")
def stats():
    try:
        return {"status": "success", "total_records": con.execute("SELECT COUNT(*) FROM users").fetchone()[0]}
    except Exception:
        logger.exception("stats failed")
        return {"status": "error"}

@app.get("/health")
def health():
    try:
        con.execute("SELECT 1").fetchone()
        return {"status": "healthy", "database": "connected"}
    except Exception:
        return {"status": "unhealthy"}
