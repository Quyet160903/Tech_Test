"""Content models for the Knowledge Base System"""

from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel


class ContentItem(BaseModel):
    """A content item extracted from a source"""
    id: str
    text: str
    title: str
    url: str
    source_id: str
    metadata: Dict[str, Any]
    timestamp: datetime
    
    class Config:
        arbitrary_types_allowed = True 