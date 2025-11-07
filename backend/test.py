import sqlite3
import os
import json
import numpy as np
import faiss
from deepface import DeepFace

DB_PATH = "children.db"
UPLOAD_DIR = "uploads"
INDEX_PATH = "faiss_index.index"
ID_MAP_PATH = "id_map.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load old records
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT name, age, contact, image_path FROM children")
records = c.fetchall()
conn.close()

# Drop old table and recreate with proper schema
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("DROP TABLE IF EXISTS children")
c.execute("""
CREATE TABLE children (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    contact TEXT,
    image_path TEXT,
    processed INTEGER DEFAULT 0,
    status TEXT DEFAULT 'missing'
)
""")
conn.commit()

# Re-insert previous records
id_map = {}
dim = 512
index = faiss.IndexFlatL2(dim)


# Save FAISS index and ID map
faiss.write_index(index, INDEX_PATH)
with open(ID_MAP_PATH, "w") as f:
    json.dump(id_map, f)

print(f"Seeded {len(records)} records, FAISS index updated, database schema fixed.")
