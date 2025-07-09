"""Knowledge organization package for the Knowledge Base System"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pydantic import BaseModel


class KnowledgeConfig(BaseModel):
    """Configuration for the knowledge organization system"""
    entity_types: Optional[List[str]] = None
    relationship_types: Optional[List[str]] = None
    taxonomy_file: Optional[str] = None
    auto_link: bool = True
    confidence_threshold: float = 0.7


class EntityResult(BaseModel):
    """Result of an entity operation"""
    success: bool
    entity_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RelationshipResult(BaseModel):
    """Result of a relationship operation"""
    success: bool
    relationship_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseKnowledgeOrganizer(ABC):
    """Base class for knowledge organization systems"""
    
    @abstractmethod
    async def initialize(self, config: KnowledgeConfig) -> bool:
        """Initialize the knowledge organization system"""
        pass
    
    @abstractmethod
    async def extract_entities(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities from content"""
        pass
    
    @abstractmethod
    async def extract_relationships(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        pass
    
    @abstractmethod
    async def add_entity(self, entity: Dict[str, Any]) -> EntityResult:
        """Add an entity to the knowledge base"""
        pass
    
    @abstractmethod
    async def add_relationship(self, relationship: Dict[str, Any]) -> RelationshipResult:
        """Add a relationship to the knowledge base"""
        pass
    
    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get an entity from the knowledge base"""
        pass
    
    @abstractmethod
    async def find_related_entities(self, entity_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Find entities related to the given entity"""
        pass 