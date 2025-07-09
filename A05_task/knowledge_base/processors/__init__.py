"""Processors package for the Knowledge Base System"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pydantic import BaseModel


class ProcessorConfig(BaseModel):
    """Configuration for a content processor"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    filters: Optional[Dict[str, Any]] = None


class ProcessorResult(BaseModel):
    """Result of a content processing operation"""
    content_chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    status: str
    error: Optional[str] = None


class BaseProcessor(ABC):
    """Base class for all content processors"""
    
    @abstractmethod
    async def process(self, content: List[Dict[str, Any]], config: ProcessorConfig) -> ProcessorResult:
        """Process content from an extractor"""
        pass 