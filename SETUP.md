## Mini-Project Local Setup Instructions

### 1. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 2. Install Python packages

```bash
pip install -r requirements.txt
```

### 3. Install Next.js Dashboard Dependencies

```bash
cd dashboard
npm install
cd ..
```

### 4. Start FastAPI Server

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### 5. Start Next.js Dashboard (New Terminal)

```bash
cd dashboard
npm run dev
```

The dashboard will be available at `http://localhost:3000`

**Default Login Credentials:**

* Username: `admin`
* Password: `admin123`

You can modify these credentials in
`dashboard/src/lib/auth.ts`

### 6. Start Real-time Recognition Module (New Terminal)

```bash
python core/realtime_recognition.py --camera 0
```

**Camera Options:**

* `0` for default webcam
* IP/URL for external device camera (e.g., `http://192.168.1.100:8080/video`)

### 7. Project Structure

```
Mini-Project/
├── backend/
│   └── main.py              # FastAPI server
│   └── uploads/             # Image storage
├── core/
│   └── realtime_recognition.py  # Face recognition
├── dashboard/               # Next.js dashboard
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── lib/
│   ├── package.json
│   └── next.config.js
├── children.db              # SQLite database
├── faiss_index.index        # Face embeddings
├── id_map.json              # ID mappings
└── requirements.txt
```

### 8. Important Notes

* **Run all processes concurrently** in separate terminals
* The `uploads/`, `faiss_index.index`, and `id_map.json` files will be created automatically if missing
* FastAPI must run on **port 8000** for dashboard connection
* WebSocket notifications work in real time when all services are active
* For external device cameras, pass the IP stream URL to `--camera`

### 9. Dashboard Features

The Next.js dashboard includes:

* ✅ **Admin Authentication** – Secure login system (credentials in `src/lib/auth.ts`)
* ✅ **Real-time WebSocket Notifications** – Instant alerts when children are found
* ✅ **Dark Mode** – Sleek UI with dark theme by default
* ✅ **Multi-page Navigation** – Overview, Add Child, Recently Found, All Records, Notifications
* ✅ **Statistics Dashboard** – Visual cards showing key metrics
* ✅ **Responsive Design** – Works seamlessly on desktop and mobile

### 10. Quick Start (All-in-One)

Run these commands in separate terminals:

**Terminal 1 – Backend**

```bash
venv\Scripts\activate && uvicorn backend.main:app --reload
```

**Terminal 2 – Dashboard**

```bash
cd dashboard && npm run dev
```

**Terminal 3 – Recognition**

```bash
venv\Scripts\activate && python core/realtime_recognition.py --camera 0
```

### 11. Troubleshooting

**Dashboard won't start:**

```bash
cd dashboard
rm -rf .next node_modules package-lock.json
npm install
npm run dev
```

**CORS errors:**

* Ensure FastAPI has CORS middleware for `http://localhost:3000`

**WebSocket not connecting:**

* Confirm FastAPI runs on `http://127.0.0.1:8000`
* Verify the WebSocket endpoint: `ws://127.0.0.1:8000/ws/notifications`

**Images not loading:**

* Ensure FastAPI has static file mounting configured
* Confirm `uploads/` directory exists

