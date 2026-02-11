# Installation

## Option 1: Docker Compose (Recommended)

1. Create `backend/.env` based on `backend/.env.example` and set the API keys you plan to use.
2. From the repo root, build and start the stack:

```bash
docker compose up --build
```

3. Open the services:

- Frontend UI: http://localhost (or https://localhost if you set up TLS)
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

## HTTPS Without a Domain (Public IP or Local)

- Local/dev TLS (mkcert): see `frontend/certs/README.md`.
- Public IP TLS (short-lived IP certs): see `frontend/certs/README.md`.

## Option 2: Manual Development Setup

**Backend**

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

3. Create `backend/.env` based on `backend/.env.example`.
4. Run the API server:

```bash
python run_server.py
```

**Frontend**

1. Install dependencies and start the dev server:

```bash
cd frontend
npm install
npm run dev
```

2. Open the UI at http://localhost:3000

## Notes

- Generated artifacts are saved under `outputs/` (embeddings, models, labeled CSVs).
- For detailed API endpoints, see `backend/API_DOCUMENTATION.md`.
