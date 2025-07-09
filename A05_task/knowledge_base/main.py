"""
Main application entry point for the Knowledge Base System
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from knowledge_base.utils.logging import logger

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Knowledge Base API",
    description="Multi-Source Knowledge Base with AI Agent",
    version="0.1.0",
)

# Add CORS middleware (minimal for Gradio if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Gradio will run on different port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import API routers
from knowledge_base.api.routes import router as api_router

# Include routers
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    """Root endpoint returning system status"""
    logger.info("Root endpoint accessed")
    return {
        "status": "online",
        "name": "Knowledge Base System",
        "version": "0.1.0",
    }

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("knowledge_base.main:app", host=host, port=port, reload=True) 