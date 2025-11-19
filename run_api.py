#!/usr/bin/env python3
"""
Simple script to run the Gemäldeagent API server.

Usage:
    python run_api.py
"""
import uvicorn

from backend.core.config import settings

if __name__ == "__main__":
    print("=" * 80)
    print("Starting Gemäldeagent API Server")
    print("=" * 80)
    print(f"Host: {settings.API_HOST}")
    print(f"Port: {settings.API_PORT}")
    print(f"Docs: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print("=" * 80)

    uvicorn.run(
        "backend.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        workers=settings.API_WORKERS,
        log_level="info"
    )
