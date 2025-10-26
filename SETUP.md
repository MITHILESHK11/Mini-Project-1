# Mini-Project Local Setup Instructions

1. **Create and activate a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

2. **Install required packages**

```bash
pip install -r requirements.txt
```

3. **Start FastAPI server**

```bash
uvicorn backend.main:app --reload
```

4. **Start real-time recognition module**

```bash
python core/realtime_recognition.py --camera 0
```

* Use `--camera <source>` to change camera:

  * `0` for default webcam
  * IP/URL for external device camera

5. **Start dashboard**

```bash
streamlit run dashboard/app.py
```

6. **Notes**

* Ensure `uploads/`, `faiss_index.index`, and `id_map.json` exist or will be created automatically.
* Start each process in a separate terminal.
* For external device camera, pass the IP stream URL to `--camera`.

This will run **FastAPI**, **recognition**, and **dashboard** concurrently, allowing real-time face recognition with alerts.
