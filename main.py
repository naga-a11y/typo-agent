import os
import logging
from fastapi import FastAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Minimal FastAPI app ---
app = FastAPI(title="FAQ API Test", version="1.0.0")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "API is working", "status": "ok"}

@app.get("/healthz")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/test")
async def test():
    """Test endpoint"""
    return {
        "message": "Test successful",
        "port": os.environ.get("PORT", "not set"),
        "env_vars": {
            "PROJECT_ID": os.environ.get("PROJECT_ID", "not set"),
            "TOOLBOX_URL": os.environ.get("TOOLBOX_URL", "not set")
        }
    }

# --- Entry point ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
