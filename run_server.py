"""
Server Startup Script
Convenient script to run the FastAPI server
"""
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get configuration from environment or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print("=" * 70)
    print("Starting Hijab Detection API Server")
    print("=" * 70)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"API Docs: http://localhost:{port}/docs")
    print(f"ReDoc: http://localhost:{port}/redoc")
    print("=" * 70)
    
    uvicorn.run(
        "services.main:app",  # <-- point to your FastAPI app
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
