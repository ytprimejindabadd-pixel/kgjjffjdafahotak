from fastapi import FastAPI, HTTPException, Query
import duckdb
from contextlib import asynccontextmanager
import os

con = duckdb.connect()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🟢 DuckDB extensions load ho rahi hain...")
    con.execute("PRAGMA memory_limit='400MB'")
    con.execute("PRAGMA threads=2")
    con.execute("SET enable_object_cache=true")
    con.execute("SET preserve_insertion_order=false")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    
    PARQUET_URL = "https://huggingface.co/datasets/tfqdeadlo/Inddatainonefile/resolve/main/users_data.parquet"
    
    con.execute(f"CREATE OR REPLACE VIEW users_view AS SELECT * FROM read_parquet('{PARQUET_URL}')")
    print("🟢 View ready!")
    yield
    con.close()

app = FastAPI(title="Super Fast Search API", lifespan=lifespan)

@app.get("/")
def home():
    return {"message": "Fast API online hai! Use /search?mobile=YOUR_NUMBER"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/search")
def search_mobile(  # ✅ async hata diya
    mobile: str | None = Query(None),
    q: str | None = Query(None),
    num: str | None = Query(None),
    number: str | None = Query(None),
    phone: str | None = Query(None),
    aadhar: str | None = Query(None),
    aadhaar: str | None = Query(None),
):
    target = (q or mobile or num or number or phone or aadhar or aadhaar or "").strip()
    if not target:
        raise HTTPException(status_code=422, detail="Enter mobile number to search")
    
    try:
        query = """
            SELECT mobile, name, fname, address, alt, circle, id, email
            FROM users_view WHERE mobile = ? LIMIT 1
        """
        result = con.execute(query, [str(target)]).fetchall()
        
        if not result and target.isdigit() and len(target) == 12:
            query2 = """
                SELECT mobile, name, fname, address, alt, circle, id, email
                FROM users_view WHERE id = ? LIMIT 1
            """
            result = con.execute(query2, [str(target)]).fetchall()
        
        if not result:
            return {"status": "error", "message": f"Mobile number {target} nahi mila."}
        
        columns = [desc[0] for desc in con.description]
        row_dict = dict(zip(columns, result[0]))
        return {"status": "success", "data": row_dict}
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail="Query timeout ya process nahi ho payi.")


# ✅ Render ke liye zaroori
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
