"""Local development server: `uv run python -m app` (auto-reload on code changes)."""

import uvicorn

if __name__ == "__main__":
    print("📍 API docs available at: http://localhost:8000/docs")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
