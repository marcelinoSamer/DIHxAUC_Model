#!/usr/bin/env python3
"""
Render entry point for FlavorFlow Craft API
"""
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import the FastAPI app
from api.main import app

# For Render, the app variable needs to be available at module level
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)