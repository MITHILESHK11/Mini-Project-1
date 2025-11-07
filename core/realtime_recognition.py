import cv2
import numpy as np
import faiss
import sqlite3
import json
import threading
from deepface import DeepFace
import os
import time
import argparse
import requests

# --- Config ---
DB_PATH = "children.db"
INDEX_PATH = "faiss_index.index"
ID_MAP_PATH = "id_map.json"
MODEL_NAME = "Facenet"
THRESHOLD = 0.6  # Use same threshold as backend
BACKEND_NOTIFY_URL = "http://127.0.0.1:8000/notify_found/"

def l2_normalize(x):
    return x / np.linalg.norm(x)

def build_faiss_index():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT rowid, image_path FROM children")
    records = c.fetchall()
    conn.close()

    embeddings = []
    id_map = {}
    for rowid, image_path in records:
        try:
            rep = DeepFace.represent(img_path=image_path, model_name=MODEL_NAME, enforce_detection=False)
            if not rep:
                continue
            emb = l2_normalize(np.array(rep[0]["embedding"], dtype=np.float32))
            embeddings.append(emb)
            id_map[str(len(id_map))] = rowid
        except Exception as e:
            print(f"Error processing ID {rowid}: {e}")

    if embeddings:
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))
        faiss.write_index(index, INDEX_PATH)
        with open(ID_MAP_PATH, "w") as f:
            json.dump(id_map, f)
        print(f"FAISS index built with {index.ntotal} embeddings")
    else:
        index = None
        print("No embeddings found in DB.")
    return index, id_map

def match_embedding(new_emb, index, id_map, threshold=THRESHOLD):
    if index is None or new_emb is None:
        return None
    new_emb = l2_normalize(np.array(new_emb, dtype=np.float32)).reshape(1, -1)
    D, I = index.search(new_emb, 1)
    dist, idx = D[0][0], I[0][0]
    print(f"Distance to nearest: {dist:.4f}")
    if dist < threshold:
        matched_id = id_map[str(idx)]
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name, age, contact, image_path, status FROM children WHERE rowid=?", (matched_id,))
        result = c.fetchone()
        conn.close()
        if result:
            return {
                "matched": True,
                "id": matched_id,
                "name": result[0],
                "age": result[1],
                "contact": result[2],
                "image": result[3],
                "distance": float(dist),
                "status": result[4]
            }
    return {"matched": False}

def notify_backend_found(child_id, distance):
    data = {"child_id": child_id, "distance": distance}
    try:
        resp = requests.post(BACKEND_NOTIFY_URL, json=data)
        if resp.status_code == 200:
            print("Backend notified:", resp.json())
        else:
            print("Backend notification failed:", resp.status_code)
    except Exception as e:
        print("Exception notifying backend:", e)

def start_recognition(camera_source=0):
    # Load index and map
    if os.path.exists(INDEX_PATH) and os.path.exists(ID_MAP_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(ID_MAP_PATH) as f:
            id_map = json.load(f)
        print(f"Loaded FAISS index with {index.ntotal} embeddings")
    else:
        index, id_map = build_faiss_index()

    cap = cv2.VideoCapture(camera_source)
    already_detected = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.astype(np.uint8)

        try:
            rep = DeepFace.represent(frame_rgb, model_name=MODEL_NAME, enforce_detection=False)
            if rep:
                new_emb = np.array(rep[0]["embedding"], dtype=np.float32)
                match = match_embedding(new_emb, index, id_map)
                if match["matched"] and match["id"] not in already_detected:
                    print(f"Detected: {match['name']} | Distance: {match['distance']:.4f} | Status: {match['status']}")
                    already_detected.add(match["id"])
                    notify_backend_found(match["id"], match["distance"])
                    name_display = f"{match['name']} ({match['status']})"
                else:
                    name_display = "Unknown"
            else:
                name_display = "Unknown"
        except Exception as e:
            print(f"Error generating embedding: {e}")
            name_display = "Unknown"

        cv2.putText(frame, name_display, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.imshow("Live Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start real-time face recognition.")
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Camera source. Use 0 for default webcam or IP stream URL."
    )
    args = parser.parse_args()
    camera_source = int(args.camera) if args.camera.isdigit() else args.camera
    start_recognition(camera_source=camera_source)
