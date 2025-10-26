from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import sqlite3, os, json, threading
import numpy as np
import faiss
from deepface import DeepFace

app = FastAPI()

DB_PATH = "children.db"
UPLOAD_DIR = "uploads"
INDEX_PATH = "faiss_index.index"
ID_MAP_PATH = "id_map.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)
lock = threading.Lock()

# --- DB Setup ---
with lock:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS children (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        contact TEXT,
        image_path TEXT
    )
    """)
    conn.commit()
    conn.close()

# --- FAISS Setup ---
DIM = 128  # Facenet embedding
if os.path.exists(INDEX_PATH) and os.path.exists(ID_MAP_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(ID_MAP_PATH) as f:
        id_map = json.load(f)
    id_counter = max([int(k) for k in id_map.keys()]) + 1 if id_map else 0
else:
    index = faiss.IndexFlatL2(DIM)
    id_map = {}
    id_counter = 0

def l2_normalize(x):
    return x / np.linalg.norm(x)

def async_update_index(image_path, rowid):
    """Update FAISS index in a separate thread."""
    def worker():
        global index, id_map
        try:
            emb = np.array(
                DeepFace.represent(image_path, model_name="Facenet", enforce_detection=False)[0]["embedding"],
                dtype=np.float32
            )
            emb = l2_normalize(emb)
            if emb.shape[0] != index.d:
                print(f"Embedding dimension {emb.shape[0]} != index dimension {index.d}, skipping ID {rowid}")
                return
            with lock:
                index.add(np.array([emb]))
                id_map[str(len(id_map))] = rowid
                faiss.write_index(index, INDEX_PATH)
                with open(ID_MAP_PATH, "w") as f:
                    json.dump(id_map, f)
                print(f"Added ID {rowid} to FAISS index. Total embeddings: {index.ntotal}")
        except Exception as e:
            print(f"Error processing embedding for ID {rowid}: {e}")
    threading.Thread(target=worker, daemon=True).start()

@app.post("/add_child/")
async def add_child(name: str = Form(...), age: int = Form(...), contact: str = Form(...), image: UploadFile = None):
    img_path = os.path.join(UPLOAD_DIR, image.filename)
    with open(img_path, "wb") as f:
        f.write(await image.read())
    print(f"Saved image: {img_path}")

    with lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        c.execute("INSERT INTO children (name, age, contact, image_path) VALUES (?, ?, ?, ?)", (name, age, contact, img_path))
        conn.commit()
        rowid = c.lastrowid
        conn.close()
        print(f"Inserted record ID {rowid}")

    async_update_index(img_path, rowid)
    return JSONResponse({"status": "success", "message": f"{name} added", "id": rowid})

@app.get("/get_children/")
def get_children():
    with lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT id, name, age, contact, image_path FROM children")
        rows = c.fetchall()
        conn.close()
    return [{"id": i, "name": n, "age": a, "contact": c_, "image": img} for i,n,a,c_,img in rows] if rows else []

@app.post("/match_child/")
async def match_child(image: UploadFile):
    img_path = os.path.join(UPLOAD_DIR, f"temp_{image.filename}")
    with open(img_path, "wb") as f:
        f.write(await image.read())

    try:
        emb = np.array(
            DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"],
            dtype=np.float32
        )
        emb = l2_normalize(emb).reshape(1, -1)
    except Exception:
        os.remove(img_path)
        return {"matched": False, "distance": None}

    with lock:
        D, I = index.search(emb, 1)
        dist, idx = D[0][0], I[0][0]
        if dist < 0.6:  # normalized embeddings threshold
            matched_id = id_map.get(str(idx))
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            c = conn.cursor()
            c.execute("SELECT id, name, age, contact, image_path FROM children WHERE id=?", (matched_id,))
            result = c.fetchone()
            conn.close()
            os.remove(img_path)
            return {"matched": True, "distance": float(dist), "child": {"id": result[0], "name": result[1], "age": result[2], "contact": result[3], "image": result[4]}}
        else:
            os.remove(img_path)
            return {"matched": False, "distance": float(dist)}
