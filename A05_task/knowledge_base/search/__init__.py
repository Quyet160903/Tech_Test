"""Search package for the Knowledge Base System"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pydantic import BaseModel


class SearchConfig(BaseModel):
    """Configuration for a search engine"""
    max_results: int = 10
    min_relevance: float = 0.6
    include_metadata: bool = True
    include_content: bool = True


class SearchResult(BaseModel):
    """Result of a search operation"""
    items: List[Dict[str, Any]]
    total_count: int
    metadata: Dict[str, Any]
    error: Optional[str] = None


class BaseSearch(ABC):
    """Base class for all search engines"""
    
    @abstractmethod
    async def initialize(self, config: SearchConfig) -> bool:
        """Initialize the search engine"""
        pass
    
    @abstractmethod
    async def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> SearchResult:
        """Search for documents matching the query"""
        pass
    
    @abstractmethod
    async def browse(self, category: str, page: int = 1, page_size: int = 10) -> SearchResult:
        """Browse documents by category"""
        pass
    
    @abstractmethod
    async def suggest(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """Suggest query completions"""
        pass 