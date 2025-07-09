"""Extractors package for the Knowledge Base System"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pydantic import BaseModel


class ExtractorConfig(BaseModel):
    """Configuration for an extractor"""
    source_id: str
    max_pages: int = 100
    max_depth: int = 3
    follow_links: bool = True
    headers: Optional[Dict[str, str]] = None
    filters: Optional[Dict[str, Any]] = None


class ExtractorResult(BaseModel):
    """Result of an extraction process"""
    source_id: str
    content: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    status: str
    error: Optional[str] = None


class BaseExtractor(ABC):
    """Base class for all extractors"""
    
    @abstractmethod
    async def extract(self, config: ExtractorConfig) -> ExtractorResult:
        """Extract content from a source"""
        pass
    
    @abstractmethod
    async def validate_source(self, source_url: str) -> bool:
        """Validate if a source can be processed by this extractor"""
        pass 