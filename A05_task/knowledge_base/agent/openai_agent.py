"""OpenAI agent for the Knowledge Base System"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from knowledge_base.agent import BaseAgent, AgentConfig, AgentResponse
from knowledge_base.storage.vector_storage import VectorStorage
from knowledge_base.storage.mongodb_storage import MongoDBStorage
from knowledge_base.storage.memory_storage import MemoryStorage


class OpenAIAgent(BaseAgent):
    """OpenAI-based agent for the knowledge base"""
    
    def __init__(self, vector_storage: Optional[VectorStorage] = None, mongo_storage: Optional[MongoDBStorage] = None, memory_storage: Optional[MemoryStorage] = None):
        self.config = None
        self.llm = None
        self.vector_storage = vector_storage
        self.mongo_storage = mongo_storage
        self.memory_storage = memory_storage
        self.system_prompt = None
    
    async def initialize(self, config: AgentConfig) -> bool:
        """Initialize the OpenAI agent"""
        try:
            self.config = config
            
            # Get API key from config or environment
            api_key = config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("OpenAI API key not found")
                return False
            
            # Initialize the language model
            self.llm = ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                openai_api_key=api_key,
            )
            
            # Set up the system prompt
            self.system_prompt = config.system_prompt or self._get_default_system_prompt()
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize OpenAI agent: {str(e)}")
            return False
    
    async def query(self, query_text: str, filters: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process a query and generate a response"""
        if not self.llm:
            return AgentResponse(
                answer="Agent not initialized",
                sources=[],
                related_entities=[],
                confidence=0.0,
                metadata={},
                error="Agent not initialized",
            )
        
        try:
            # Generate context for the query
            context_docs = await self.generate_context(query_text, filters)
            
            # Extract sources and related entities
            sources = []
            related_entities = []
            
            for doc in context_docs:
                # Add to sources
                source_info = {
                    "id": doc.get("id", ""),
                    "title": doc.get("metadata", {}).get("title", ""),
                    "url": doc.get("metadata", {}).get("url", ""),
                    "relevance_score": doc.get("relevance_score", 0.0),
                }
                sources.append(source_info)
                
                # Add any related entities
                if "entities" in doc:
                    for entity in doc["entities"]:
                        related_entities.append({
                            "id": entity.get("id", ""),
                            "name": entity.get("name", ""),
                            "entity_type": entity.get("entity_type", ""),
                            "relevance": entity.get("confidence", 0.0),
                        })
            
            # Prepare the context for the prompt
            context_text = self._prepare_context_text(context_docs)
            
            # Create the messages directly instead of using ChatPromptTemplate
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"Context information:\n{context_text}\n\nQuery: {query_text}")
            ]
            
            # Generate the response
            response = await self.llm.ainvoke(messages)
            
            # Extract the answer
            answer = response.content
            
            # Calculate confidence based on context availability and quality
            confidence = self._calculate_confidence(context_docs, sources)
            
            return AgentResponse(
                answer=answer,
                sources=sources[:5],  # Limit to top 5 sources
                related_entities=self._deduplicate_entities(related_entities)[:5],  # Limit to top 5 entities
                confidence=confidence,
                metadata={
                    "query_time": datetime.now().isoformat(),
                    "model": self.config.model_name,
                    "context_docs_count": len(context_docs),
                },
            )
            
        except Exception as e:
            return AgentResponse(
                answer=f"Error processing query: {str(e)}",
                sources=[],
                related_entities=[],
                confidence=0.0,
                metadata={},
                error=str(e),
            )
    
    async def generate_context(self, query_text: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate context for a query by searching both vector and mongo storage"""
        all_results = []
        
        try:
            # Prepare the query
            query = {"text": query_text}
            
            # Only add filters if they exist and are not empty
            if filters and any(v is not None and v != "" for v in filters.values()):
                query["filter"] = filters
            
            # Try vector storage first (ChromaDB)
            if self.vector_storage:
                try:
                    vector_results = await self.vector_storage.retrieve(query, limit=10)
                    all_results.extend(vector_results)
                except Exception as e:
                    print(f"Vector storage search failed: {str(e)}")
            
            # Try MongoDB storage
            if self.mongo_storage:
                try:
                    mongo_results = await self.mongo_storage.retrieve(query, limit=10)
                    all_results.extend(mongo_results)
                except Exception as e:
                    print(f"MongoDB storage search failed: {str(e)}")
            
            # Fallback to memory storage if no results
            if not all_results and self.memory_storage:
                try:
                    memory_results = await self.memory_storage.retrieve(query, limit=10)
                    all_results.extend(memory_results)
                except Exception as e:
                    print(f"Memory storage search failed: {str(e)}")
            
            # Deduplicate results by content hash or ID
            unique_results = []
            seen_ids = set()
            seen_content = set()
            
            for result in all_results:
                result_id = result.get("id", "")
                content_hash = hash(result.get("text", ""))
                
                if result_id and result_id not in seen_ids:
                    seen_ids.add(result_id)
                    unique_results.append(result)
                elif content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append(result)
            
            # Sort by relevance score
            unique_results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
            
            # Return top 10 results
            return unique_results[:10]
            
        except Exception as e:
            print(f"Error generating context: {str(e)}")
            return []
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the agent"""
        return """You are a helpful AI assistant for a knowledge base system.
Your task is to provide accurate, helpful answers based on the context information provided.
If the answer is not in the context, say that you don't know and suggest what information might help.
Always cite your sources by referring to the titles or URLs provided in the context.
Keep your answers concise and focused on the query."""
    
    def _prepare_context_text(self, context_docs: List[Dict[str, Any]]) -> str:
        """Prepare context text from documents"""
        if not context_docs:
            return "No relevant context found."
        
        context_parts = []
        
        for i, doc in enumerate(context_docs):
            # Extract document information
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            title = metadata.get("title", f"Document {i+1}")
            url = metadata.get("url", "")
            relevance_score = doc.get("relevance_score", 0.0)
            
            # Format the document
            doc_text = f"--- Document {i+1}: {title} ---\n"
            if url:
                doc_text += f"Source: {url}\n"
            if relevance_score > 0:
                doc_text += f"Relevance: {relevance_score:.2f}\n"
            doc_text += f"{text}\n"
            
            context_parts.append(doc_text)
        
        return "\n".join(context_parts)
    
    def _calculate_confidence(self, context_docs: List[Dict[str, Any]], sources: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on context quality and availability"""
        if not context_docs:
            return 0.2  # Low confidence when no context available
        
        # Base confidence on number and quality of context documents
        doc_count_score = min(len(context_docs) / 5.0, 1.0)  # Max score when 5+ docs
        
        # Average relevance score of top documents
        relevance_scores = [doc.get("relevance_score", 0.0) for doc in context_docs[:3]]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        # Combine scores
        confidence = (doc_count_score * 0.4 + avg_relevance * 0.6)
        
        # Ensure confidence is between 0.1 and 0.95
        return max(0.1, min(0.95, confidence))
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate entities by ID"""
        unique_entities = {}
        for entity in entities:
            entity_id = entity.get("id", "")
            if entity_id and (entity_id not in unique_entities or entity.get("relevance", 0.0) > unique_entities[entity_id].get("relevance", 0.0)):
                unique_entities[entity_id] = entity
        
        return list(unique_entities.values())