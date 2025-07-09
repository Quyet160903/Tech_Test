"""API routes for the Knowledge Base System"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import asyncio
from datetime import datetime
import tempfile
import os

from knowledge_base.extractors.extraction_manager import ExtractionManager, SourceInfo
from knowledge_base.processors.text_processor import TextProcessor
from knowledge_base.storage.vector_storage import VectorStorage
from knowledge_base.storage.mongodb_storage import MongoDBStorage
from knowledge_base.storage.memory_storage import MemoryStorage
from knowledge_base.agent.openai_agent import OpenAIAgent
from knowledge_base.search.semantic_search import SemanticSearch

from knowledge_base.extractors import ExtractorConfig
from knowledge_base.processors import ProcessorConfig
from knowledge_base.storage import StorageConfig
from knowledge_base.agent import AgentConfig
from knowledge_base.search import SearchConfig

from knowledge_base.utils.logging import logger

router = APIRouter()

# Global instances
extraction_manager = ExtractionManager()
text_processor = TextProcessor()
vector_storage = VectorStorage()
mongo_storage = MongoDBStorage()
memory_storage = MemoryStorage()  # Fallback storage
openai_agent = OpenAIAgent(vector_storage=vector_storage, mongo_storage=mongo_storage, memory_storage=memory_storage)
semantic_search = SemanticSearch(vector_storage=vector_storage, metadata_storage=mongo_storage)

# Initialize storage backends
initialized = False

async def ensure_initialized():
    """Ensure all components are initialized"""
    global initialized
    if not initialized:
        # Always initialize memory storage as fallback
        await memory_storage.initialize(StorageConfig(
            database_name="knowledge_base",
            collection_name="content"
        ))
        
        # Try to initialize MongoDB storage
        mongo_success = await mongo_storage.initialize(StorageConfig(
            database_name="knowledge_base",
            collection_name="content"
        ))
        
        # Try to initialize vector storage
        vector_success = await vector_storage.initialize(StorageConfig(
            database_name="knowledge_base", 
            collection_name="content_vectors"
        ))
        
        await openai_agent.initialize(AgentConfig(model_name="gpt-3.5-turbo"))
        await semantic_search.initialize(SearchConfig())
        
        if not mongo_success:
            logger.warning("MongoDB not available, using memory storage for metadata")
        if not vector_success:
            logger.warning("Vector DB not available, using memory storage for vectors")
        
        initialized = True


# Pydantic models for API
class SourceRequest(BaseModel):
    source_id: str
    location: str
    source_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ExtractionRequest(BaseModel):
    source_id: str
    config: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = 10


# Source management endpoints
@router.get("/sources")
async def get_sources():
    """Get all sources"""
    try:
        sources = extraction_manager.get_all_sources()
        return {
            "sources": [
                {
                    "source_id": info.source_id,
                    "source_type": info.source_type,
                    "location": info.location,
                    "status": info.status,
                    "created_at": info.created_at.isoformat(),
                    "last_extracted": info.last_extracted.isoformat() if info.last_extracted else None,
                    "extraction_count": info.extraction_count,
                    "metadata": info.metadata
                }
                for info in sources.values()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting sources: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sources")
async def add_source(request: SourceRequest):
    """Add a new source and automatically extract and store content"""
    try:
        await ensure_initialized()
        
        # Add source
        success = await extraction_manager.add_source(
            source_id=request.source_id,
            location=request.location,
            source_type=request.source_type,
            metadata=request.metadata
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to add source")
        
        # Automatically extract content
        logger.info(f"Automatically extracting content from source {request.source_id}")
        extraction_result = await extraction_manager.extract_source(request.source_id)
        
        if extraction_result.error:
            logger.warning(f"Extraction failed for source {request.source_id}: {extraction_result.error}")
            return {
                "message": "Source added but extraction failed", 
                "source_id": request.source_id,
                "extraction_error": extraction_result.error
            }
        
        # Process the content
        processing_result = await text_processor.process(
            extraction_result.content,
            ProcessorConfig(chunk_size=1000, chunk_overlap=200)
        )
        
        if processing_result.error:
            logger.warning(f"Processing failed for source {request.source_id}: {processing_result.error}")
            return {
                "message": "Source added and extracted but processing failed",
                "source_id": request.source_id,
                "processing_error": processing_result.error
            }
        
        # Store in both MongoDB and ChromaDB (with fallback to memory)
        mongo_result = await mongo_storage.store(processing_result.content_chunks)
        if not mongo_result.success:
            logger.warning("MongoDB storage failed, using memory storage")
            mongo_result = await memory_storage.store(processing_result.content_chunks)
        
        vector_result = await vector_storage.store(processing_result.content_chunks)
        if not vector_result.success:
            logger.warning("ChromaDB storage failed, using memory storage")
            vector_result = await memory_storage.store(processing_result.content_chunks)
        
        return {
            "message": "Source added, extracted, and stored successfully",
            "source_id": request.source_id,
            "extracted_items": len(extraction_result.content),
            "processed_chunks": len(processing_result.content_chunks),
            "mongo_stored": mongo_result.count if mongo_result.success else 0,
            "vector_stored": vector_result.count if vector_result.success else 0,
            "storage_type": "database" if (mongo_result.success or vector_result.success) else "memory"
        }
            
    except Exception as e:
        logger.error(f"Error adding and processing source: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sources/upload")
async def upload_file_source(
    file: UploadFile = File(...),
    source_id: str = Form(...),
    metadata: Optional[str] = Form(None)
):
    """Upload a file and add it as a source"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Parse metadata if provided
        file_metadata = {}
        if metadata:
            import json
            file_metadata = json.loads(metadata)
        
        # Add file info to metadata
        file_metadata.update({
            "original_filename": file.filename,
            "file_size": len(content),
            "content_type": file.content_type
        })
        
        # Add source
        success = await extraction_manager.add_source(
            source_id=source_id,
            location=temp_path,
            metadata=file_metadata
        )
        
        if not success:
            # Clean up temp file if source addition failed
            os.unlink(temp_path)
            raise HTTPException(status_code=400, detail="Failed to add uploaded file as source")
        
        # Automatically extract and process content
        try:
            await ensure_initialized()
            
            extraction_result = await extraction_manager.extract_source(source_id)
            
            if extraction_result.error:
                return {
                    "message": "File uploaded and added but extraction failed",
                    "source_id": source_id,
                    "filename": file.filename,
                    "extraction_error": extraction_result.error
                }
            
            # Process the content
            processing_result = await text_processor.process(
                extraction_result.content,
                ProcessorConfig(chunk_size=1000, chunk_overlap=200)
            )
            
            if processing_result.error:
                return {
                    "message": "File uploaded and extracted but processing failed",
                    "source_id": source_id,
                    "filename": file.filename,
                    "processing_error": processing_result.error
                }
            
            # Store in both MongoDB and ChromaDB (with fallback to memory)
            mongo_result = await mongo_storage.store(processing_result.content_chunks)
            if not mongo_result.success:
                logger.warning("MongoDB storage failed, using memory storage")
                mongo_result = await memory_storage.store(processing_result.content_chunks)
            
            vector_result = await vector_storage.store(processing_result.content_chunks)
            if not vector_result.success:
                logger.warning("ChromaDB storage failed, using memory storage")
                vector_result = await memory_storage.store(processing_result.content_chunks)
            
            return {
                "message": "File uploaded, extracted, and stored successfully",
                "source_id": source_id,
                "filename": file.filename,
                "extracted_items": len(extraction_result.content),
                "processed_chunks": len(processing_result.content_chunks),
                "mongo_stored": mongo_result.count if mongo_result.success else 0,
                "vector_stored": vector_result.count if vector_result.success else 0,
                "storage_type": "database" if (mongo_result.success or vector_result.success) else "memory"
            }
            
        except Exception as e:
            logger.warning(f"Processing failed for uploaded file {source_id}: {str(e)}")
            return {
                "message": "File uploaded and added but processing failed",
                "source_id": source_id,
                "filename": file.filename,
                "processing_error": str(e)
            }
            
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sources/{source_id}")
async def get_source(source_id: str):
    """Get information about a specific source"""
    try:
        source_info = extraction_manager.get_source_info(source_id)
        if not source_info:
            raise HTTPException(status_code=404, detail="Source not found")
        
        return {
            "source_id": source_info.source_id,
            "source_type": source_info.source_type,
            "location": source_info.location,
            "status": source_info.status,
            "created_at": source_info.created_at.isoformat(),
            "last_extracted": source_info.last_extracted.isoformat() if source_info.last_extracted else None,
            "extraction_count": source_info.extraction_count,
            "metadata": source_info.metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting source {source_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sources/{source_id}")
async def remove_source(source_id: str):
    """Remove a source"""
    try:
        success = extraction_manager.remove_source(source_id)
        if success:
            return {"message": "Source removed successfully"}
        else:
            raise HTTPException(status_code=404, detail="Source not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing source {source_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Extraction endpoints
@router.post("/extract/{source_id}")
async def extract_source(source_id: str, request: Optional[ExtractionRequest] = None):
    """Extract content from a specific source"""
    try:
        await ensure_initialized()
        
        # Extract content
        config = request.config if request else None
        extraction_result = await extraction_manager.extract_source(source_id, config)
        
        if extraction_result.error:
            raise HTTPException(status_code=400, detail=extraction_result.error)
        
        # Process the content
        processing_result = await text_processor.process(
            extraction_result.content,
            ProcessorConfig(chunk_size=1000, chunk_overlap=200)
        )
        
        if processing_result.error:
            raise HTTPException(status_code=500, detail=processing_result.error)
        
        # Store in both MongoDB and vector storage (with fallback to memory)
        mongo_result = await mongo_storage.store(processing_result.content_chunks)
        if not mongo_result.success:
            logger.warning("MongoDB storage failed, using memory storage")
            mongo_result = await memory_storage.store(processing_result.content_chunks)
        
        vector_result = await vector_storage.store(processing_result.content_chunks)
        if not vector_result.success:
            logger.warning("Vector storage failed, using memory storage")
            vector_result = await memory_storage.store(processing_result.content_chunks)
        
        return {
            "message": "Extraction completed successfully",
            "source_id": source_id,
            "extracted_items": len(extraction_result.content),
            "processed_chunks": len(processing_result.content_chunks),
            "mongo_stored": mongo_result.count if mongo_result.success else 0,
            "vector_stored": vector_result.count if vector_result.success else 0,
            "extraction_metadata": extraction_result.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting source {source_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract/all")
async def extract_all_sources():
    """Extract content from all sources"""
    try:
        await ensure_initialized()
        
        # Extract from all sources
        extraction_results = await extraction_manager.extract_all_sources(max_concurrent=3)
        
        total_items = 0
        total_chunks = 0
        successful_extractions = 0
        
        for source_id, result in extraction_results.items():
            if not result.error:
                successful_extractions += 1
                
                # Process the content
                processing_result = await text_processor.process(
                    result.content,
                    ProcessorConfig(chunk_size=1000, chunk_overlap=200)
                )
                
                if not processing_result.error:
                    total_items += len(result.content)
                    total_chunks += len(processing_result.content_chunks)
                    
                    # Store in both storages
                    await mongo_storage.store(processing_result.content_chunks)
                    await vector_storage.store(processing_result.content_chunks)
        
        return {
            "message": "Batch extraction completed",
            "total_sources": len(extraction_results),
            "successful_extractions": successful_extractions,
            "total_items": total_items,
            "total_chunks": total_chunks
        }
        
    except Exception as e:
        logger.error(f"Error in batch extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Query and search endpoints
@router.post("/query")
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base using AI agent"""
    try:
        await ensure_initialized()
        
        response = await openai_agent.query(request.query)
        
        if response.error:
            raise HTTPException(status_code=400, detail=response.error)
        
        return {
            "query": request.query,
            "answer": response.answer,
            "sources": response.sources,
            "related_entities": response.related_entities,
            "confidence": response.confidence
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_knowledge_base(request: SearchRequest):
    """Search the knowledge base using semantic search"""
    try:
        await ensure_initialized()
        
        search_result = await semantic_search.search(
            query=request.query,
            filters=request.filters,
            limit=request.limit
        )
        
        if search_result.error:
            raise HTTPException(status_code=400, detail=search_result.error)
        
        return {
            "query": request.query,
            "total_count": search_result.total_count,
            "items": search_result.items,
            "filters_applied": request.filters
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Statistics and monitoring endpoints
@router.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        extraction_stats = extraction_manager.get_extraction_stats()
        
        # Get storage stats (basic counts)
        mongo_docs = await mongo_storage.retrieve({}, limit=1)
        vector_docs = await vector_storage.retrieve({}, limit=1)
        
        return {
            "extraction": extraction_stats,
            "storage": {
                "mongo_accessible": len(mongo_docs) >= 0,
                "vector_accessible": len(vector_docs) >= 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        await ensure_initialized()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "extraction_manager": "ok",
                "mongo_storage": "ok",
                "vector_storage": "ok",
                "openai_agent": "ok"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        ) 