#!/usr/bin/env python3
"""
Entry point for FlavorFlow Craft API (Render / local).

Render auto-detects the `app` variable or uses the startCommand in render.yaml.
"""
import os
import sys
from pathlib import Path

# Add src to path so that `from src.*` and bare `from api.*` both resolve
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import the FastAPI app — Vercel auto-detects the `app` variable
from src.api.main import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="http://localhost", port=port)