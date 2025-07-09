"""AI agent package for the Knowledge Base System"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Configuration for an AI agent"""
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1024
    api_key: Optional[str] = None
    context_window: int = 4096
    system_prompt: Optional[str] = None


class AgentResponse(BaseModel):
    """Response from an AI agent"""
    answer: str
    sources: List[Dict[str, Any]]
    related_entities: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]
    error: Optional[str] = None


class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    @abstractmethod
    async def initialize(self, config: AgentConfig) -> bool:
        """Initialize the AI agent"""
        pass
    
    @abstractmethod
    async def query(self, query_text: str, filters: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process a query and generate a response"""
        pass
    
    @abstractmethod
    async def generate_context(self, query_text: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate context for a query"""
        pass 