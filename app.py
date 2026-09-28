from fastapi import FastAPI, Query
import duckdb
import os
import re
import threading
from contextlib import asynccontextmanager

app = FastAPI(title="User Search API")

# -------------------------------------------------
# DuckDB connection (single conn + lock = safe)
# -------------------------------------------------
con = None
db_lock = threading.Lock()

SEARCH_COLUMNS = {
    "mobile": "mobile",
    "alt": "alt",
    "id": "id",
    "email": "email",
}
VALID_TYPES = list(SEARCH_COLUMNS.keys())

PARQUET_URL = (
    "https://huggingface.co/datasets/tfqdeadlo/"
    "Inddatainonefile/resolve/main/users_data.parquet"
)


# -------------------------------------------------
# Lifespan: setup DuckDB
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global con
    print("🚀 Starting up...")
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute(
        f"CREATE OR REPLACE VIEW users AS "
        f"SELECT * FROM read_parquet('{PARQUET_URL}')"
    )
    # Warm-up query so the first request isn't slow
    try:
        con.execute("SELECT COUNT(*) FROM users").fetchone()
        print("✅ DuckDB ready")
    except Exception as e:
        print(f"⚠️ Warm-up failed: {e}")

    yield

    print("🛑 Shutting down...")
    if con:
        con.close()


app.router.lifespan_context = lifespan


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/")
def home():
    return {
        "status": "online",
        "message": "User Search API",
        "endpoints": {
            "search": "/search?type=mobile&q=9876543210&limit=100",
            "stats": "/stats",
            "health": "/health",
            "types": VALID_TYPES,
        },
    }


def _search_db(q: str, type_: str, limit: int = 100):
    try:
        col = SEARCH_COLUMNS[type_]
        query = (
            f"SELECT mobile, name, fname AS father_name, address, circle, "
            f"alt AS alternate, id AS aadhar, email "
            f"FROM users WHERE {col} = ? LIMIT ?"
        )
        with db_lock:
            results = con.execute(query, [q, limit]).fetchall()

        if not results:
            return {"success": False, "message": f"{type_} '{q}' not found"}

        columns = [
            "mobile", "name", "father_name", "address",
            "circle", "alternate", "aadhar", "email",
        ]
        rows = [dict(zip(columns, row)) for row in results]
        return {
            "success": True,
            "number": q,
            "total": len(rows),
            "results": rows,
        }
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.get("/search")
def search(
    q: str = Query(..., description="Search value"),
    type: str = Query("mobile", description=f"Search type: {VALID_TYPES}"),
    limit: int = Query(100, description="Max results to return"),
):
    q_clean = str(q).strip()
    type_clean = str(type).strip().lower()
    limit_clean = min(max(limit, 1), 500)

    if type_clean not in VALID_TYPES:
        return {"success": False, "message": f"Invalid type. Valid: {VALID_TYPES}"}

    if type_clean in ("mobile", "alt") and not re.match(r"^\d{10}$", q_clean):
        return {"success": False, "message": "Mobile number must be 10 digits"}

    if type_clean == "id" and not re.match(r"^\d{12}$", q_clean):
        return {"success": False, "message": "Aadhar ID must be 12 digits"}

    if type_clean == "email" and not re.match(r"^[^@]+@[^@]+\.[^@]+$", q_clean):
        return {"success": False, "message": "Invalid email format"}

    return _search_db(q_clean, type_clean, limit_clean)


@app.get("/stats")
def stats():
    try:
        with db_lock:
            count = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        return {"status": "success", "total_records": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/health")
def health():
    try:
        with db_lock:
            con.execute("SELECT 1").fetchone()
        return {"status": "healthy", "database": "connected"}
    except Exception:
        return {"status": "unhealthy"}


# -------------------------------------------------
# Local dev (Render uses gunicorn/uvicorn via start cmd)
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
