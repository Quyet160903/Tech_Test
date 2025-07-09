"""
Data models for the Knowledge Base System
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    """Types of data sources that can be ingested"""
    WEBSITE = "website"
    PDF = "pdf"
    API = "api"
    DATABASE = "database"
    GITHUB = "github"
    ARXIV = "arxiv"
    RSS = "rss"
    YOUTUBE = "youtube"
    DOCUMENTATION = "documentation"
    FORUM = "forum"
    QA_SITE = "qa_site"
    COURSE = "course"
    RESEARCH_PAPER = "research_paper"
    BLOG = "blog"
    WHITEPAPER = "whitepaper"
    KNOWLEDGE_GRAPH = "knowledge_graph"


class ContentType(str, Enum):
    """Types of content that can be stored"""
    TEXT = "text"
    CODE = "code"
    FORMULA = "formula"
    TABLE = "table"
    IMAGE = "image"
    DIAGRAM = "diagram"


class EntityType(str, Enum):
    """Types of entities in the knowledge base"""
    CONCEPT = "concept"
    METHOD = "method"
    ALGORITHM = "algorithm"
    TOOL = "tool"
    LIBRARY = "library"
    FRAMEWORK = "framework"
    PERSON = "person"
    ORGANIZATION = "organization"
    EVENT = "event"
    DATASET = "dataset"
    METRIC = "metric"
    TERM = "term"


class RelationshipType(str, Enum):
    """Types of relationships between entities"""
    IS_A = "is_a"
    PART_OF = "part_of"
    HAS_PART = "has_part"
    RELATED_TO = "related_to"
    PREREQUISITE = "prerequisite"
    FOLLOWS = "follows"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    USES = "uses"
    IMPLEMENTS = "implements"
    CREATED_BY = "created_by"
    APPLIES_TO = "applies_to"
    ALTERNATIVE_TO = "alternative_to"
    EXTENDS = "extends"
    CONTAINS = "contains"
    DEPENDS_ON = "depends_on"
    ENABLES = "enables"
    CONFLICTS_WITH = "conflicts_with"
    DERIVED_FROM = "derived_from"
    EQUIVALENT_TO = "equivalent_to"
    INFLUENCED_BY = "influenced_by"
    CAUSES = "causes"
    PREVENTS = "prevents"


class Source(BaseModel):
    """Data source model"""
    id: str
    name: str
    source_type: SourceType
    url: Optional[HttpUrl] = None
    description: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    credibility_score: float = 0.0
    freshness_score: float = 0.0
    authority_score: float = 0.0
    last_modified: Optional[datetime] = None
    check_frequency: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    status: str = "pending"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extraction_config: Dict[str, Any] = Field(default_factory=dict)


class Content(BaseModel):
    """Content chunk model"""
    id: str
    source_id: str
    content_type: ContentType
    text: str
    title: Optional[str] = None
    summary: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    language: Optional[str] = None
    quality_score: float = 0.0
    importance_score: float = 0.0
    readability_score: float = 0.0
    entity_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    vector: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: int = 1
    

class Entity(BaseModel):
    """Entity model for knowledge organization"""
    id: str
    name: str
    entity_type: EntityType
    description: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    definition: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    frequency: int = 1
    importance_score: float = 0.0
    quality_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    content_ids: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)
    vector: Optional[List[float]] = None


class Relationship(BaseModel):
    """Relationship between entities"""
    id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationshipType
    strength: float = 1.0
    confidence: float = 1.0
    weight: float = 1.0
    bidirectional: bool = False
    context: Optional[str] = None
    evidence: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Query(BaseModel):
    """Query model"""
    id: str
    text: str
    timestamp: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """Query result model"""
    id: str
    query_id: str
    answer: str
    sources: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.now)
    execution_time_ms: int
    metadata: Dict[str, Any] = Field(default_factory=dict) 