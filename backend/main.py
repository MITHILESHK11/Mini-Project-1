from fastapi import FastAPI, UploadFile, Form, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import sqlite3, os, json, threading, asyncio
import numpy as np
import faiss
from deepface import DeepFace

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="backend/uploads"), name="uploads")

# --- Paths and resources ---
DB_PATH = "children.db"
UPLOAD_DIR = "backend/uploads"
INDEX_PATH = os.path.join(UPLOAD_DIR, "faiss_index.index")
ID_MAP_PATH = os.path.join(UPLOAD_DIR, "id_map.json")
MODEL_NAME = "Facenet"
DIM = 128  # Embedding dimension for Facenet

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
            image_path TEXT,
            status TEXT DEFAULT 'missing'
        )
    """)
    conn.commit()
    conn.close()

# --- FAISS utility functions ---
def l2_normalize(x):
    return x / np.linalg.norm(x)

def load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(ID_MAP_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(ID_MAP_PATH) as f:
            id_map = json.load(f)
    else:
        index = faiss.IndexFlatL2(DIM)
        id_map = {}
    return index, id_map

def save_index(index, id_map):
    faiss.write_index(index, INDEX_PATH)
    with open(ID_MAP_PATH, "w") as f:
        json.dump(id_map, f)

def add_embedding_to_index(image_path, rowid):
    index, id_map = load_index()
    try:
        rep = DeepFace.represent(img_path=image_path, model_name=MODEL_NAME, enforce_detection=False)
        if not rep:
            print(f"No embedding for {image_path}")
            return
        emb = l2_normalize(np.array(rep[0]["embedding"], dtype=np.float32))
        emb = emb.reshape(1, -1)
        with lock:
            index.add(emb)
            id_map[str(len(id_map))] = rowid
            save_index(index, id_map)
        print(f"Added ID {rowid} to FAISS index. Total embeddings: {index.ntotal}")
    except Exception as e:
        print(f"Error embedding for {image_path}, ID {rowid}: {e}")

def async_update_index(image_path, rowid):
    threading.Thread(target=add_embedding_to_index, args=(image_path, rowid), daemon=True).start()

# --- API endpoints ---

@app.post("/add_child/")
async def add_child(
    name: str = Form(...),
    age: int = Form(...),
    contact: str = Form(...),
    image: UploadFile = None
):
    try:
        img_path = os.path.join(UPLOAD_DIR, image.filename)
        with open(img_path, "wb") as f:
            f.write(await image.read())
        print(f"Saved image: {img_path}")
        with lock:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            c = conn.cursor()
            c.execute(
                "INSERT INTO children (name, age, contact, image_path, status) VALUES (?, ?, ?, ?, ?)",
                (name, age, contact, img_path, 'missing')
            )
            conn.commit()
            rowid = c.lastrowid
            conn.close()
            print(f"Inserted record ID {rowid}")
        # Update FAISS index in background
        async_update_index(img_path, rowid)
        return JSONResponse({"status": "success", "message": f"{name} added", "id": rowid})
    except Exception as e:
        print(f"Error adding child: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/get_children/")
def get_children():
    with lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT id, name, age, contact, image_path, status FROM children")
        rows = c.fetchall()
        conn.close()
    return [
        {
            "id": i,
            "name": n,
            "age": a,
            "contact": c_,
            "image": img.replace("\\", "/"),  
            "status": stat
        }
        for i, n, a, c_, img, stat in rows
    ] if rows else []

# --- WebSocket for Real-Time Notifications ---
active_connections = []

@app.websocket("/ws/notifications")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    print(f"WebSocket client connected: total={len(active_connections)}")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print(f"WebSocket client disconnected: total={len(active_connections)}")

async def notify_all(message: dict):
    removals = []
    for ws in active_connections:
        try:
            await ws.send_json(message)
        except:
            removals.append(ws)
    for ws in removals:
        active_connections.remove(ws)

@app.post("/notify_found/")
async def notify_found(
    child_id: int = Body(...),
    distance: float = Body(...)
):
    with lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        c.execute("UPDATE children SET status=? WHERE id=?", ("found", child_id))
        c.execute(
            "SELECT id, name, age, contact, image_path, status FROM children WHERE id=?",
            (child_id,)
        )
        result = c.fetchone()
        conn.commit()
        conn.close()
    if result:
        asyncio.create_task(
            notify_all({
                "event": "child_found",
                "child": {
                    "id": result[0],
                    "name": result[1],
                    "age": result[2],
                    "contact": result[3],
                    "image": result[4],
                    "status": result[5],
                    "distance": distance
                }
            })
        )
        return {
            "status": "notified",
            "child": {
                "id": result[0],
                "name": result[1],
                "age": result[2],
                "contact": result[3],
                "image": result[4],
                "status": result[5],
                "distance": distance
            }
        }
    else:
        return {"status": "error", "reason": "Child not found"}
