import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("NEON_DATABASE_URL") or os.getenv("DATABASE_URL")
if not url:
    print("Error: Set NEON_DATABASE_URL or DATABASE_URL in .env")
    exit(1)

try:
    print("Connecting to Neon database...")
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    cur.execute("SELECT version();")
    print(f"Successfully connected! Version: {cur.fetchone()}")
    cur.close()
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")
