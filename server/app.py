"""
server/app.py - Entry point for the PipelineForge server.
Imports the FastAPI app from the root app.py and exposes a main() function
that can be used as a project.scripts entry point.
"""
import sys
import os

# Make sure root is on the path so imports from app.py work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from app import app  # noqa: F401 - re-export for openenv


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
