"""Storage package for the Knowledge Base System"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel


class StorageConfig(BaseModel):
    """Configuration for a storage backend"""
    connection_string: Optional[str] = None
    database_name: str
    collection_name: str
    indexes: Optional[List[Dict[str, Any]]] = None


class StorageResult(BaseModel):
    """Result of a storage operation"""
    success: bool
    document_ids: Optional[List[str]] = None
    count: Optional[int] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseStorage(ABC):
    """Base class for all storage backends"""
    
    @abstractmethod
    async def initialize(self, config: StorageConfig) -> bool:
        """Initialize the storage backend"""
        pass
    
    @abstractmethod
    async def store(self, documents: List[Dict[str, Any]]) -> StorageResult:
        """Store documents in the storage backend"""
        pass
    
    @abstractmethod
    async def retrieve(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents from the storage backend"""
        pass
    
    @abstractmethod
    async def update(self, document_id: str, updates: Dict[str, Any]) -> StorageResult:
        """Update a document in the storage backend"""
        pass
    
    @abstractmethod
    async def delete(self, query: Dict[str, Any]) -> StorageResult:
        """Delete documents from the storage backend"""
        pass 