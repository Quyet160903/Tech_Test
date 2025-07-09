"""Semantic search engine for the Knowledge Base System"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from knowledge_base.search import BaseSearch, SearchConfig, SearchResult
from knowledge_base.storage.vector_storage import VectorStorage
from knowledge_base.storage.mongodb_storage import MongoDBStorage


class SemanticSearch(BaseSearch):
    """Semantic search engine using vector embeddings"""
    
    def __init__(self, vector_storage: Optional[VectorStorage] = None, metadata_storage: Optional[MongoDBStorage] = None):
        self.config = None
        self.vector_storage = vector_storage
        self.metadata_storage = metadata_storage
        self.recent_queries = []
        self.popular_categories = {}
    
    async def initialize(self, config: SearchConfig) -> bool:
        """Initialize the semantic search engine"""
        try:
            self.config = config
            return True
            
        except Exception as e:
            print(f"Failed to initialize semantic search: {str(e)}")
            return False
    
    async def search(self, query: str, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> SearchResult:
        """Search for documents matching the query"""
        if not self.vector_storage:
            return SearchResult(
                items=[],
                total_count=0,
                metadata={"error": "Vector storage not initialized"},
                error="Vector storage not initialized",
            )
        
        try:
            # Track the query for analytics
            self._track_query(query)
            
            # Prepare the query for vector search
            vector_query = {"text": query}
            
            # Only add filters if they exist and are not empty
            if filters and any(v is not None and v != "" for v in filters.values()):
                vector_query["filter"] = filters
            
            # Use the provided limit or fall back to config max_results
            search_limit = limit if limit is not None else self.config.max_results
            
            # Search for relevant documents
            results = await self.vector_storage.retrieve(vector_query, limit=search_limit)
            
            # Filter by minimum relevance
            filtered_results = [
                doc for doc in results 
                if doc.get("relevance_score", 0.0) >= self.config.min_relevance
            ]
            
            # Format the results
            items = []
            for doc in filtered_results:
                item = {
                    "id": doc.get("id", ""),
                    "relevance_score": doc.get("relevance_score", 0.0),
                }
                
                # Include metadata if configured
                if self.config.include_metadata:
                    item["metadata"] = doc.get("metadata", {})
                
                # Include content if configured
                if self.config.include_content:
                    item["text"] = doc.get("text", "")
                
                items.append(item)
            
            return SearchResult(
                items=items,
                total_count=len(items),
                metadata={
                    "query": query,
                    "filters": filters,
                    "limit": search_limit,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            
        except Exception as e:
            return SearchResult(
                items=[],
                total_count=0,
                metadata={"query": query},
                error=str(e),
            )
    
    async def browse(self, category: str, page: int = 1, page_size: int = 10) -> SearchResult:
        """Browse documents by category"""
        if not self.metadata_storage:
            return SearchResult(
                items=[],
                total_count=0,
                metadata={"error": "Metadata storage not initialized"},
                error="Metadata storage not initialized",
            )
        
        try:
            # Track the category browse for analytics
            self._track_category(category)
            
            # Calculate pagination
            skip = (page - 1) * page_size
            
            # Query the metadata storage
            query = {"metadata.category": category}
            results = await self.metadata_storage.retrieve(query, limit=page_size)
            
            # Format the results
            items = []
            for doc in results:
                item = {
                    "id": doc.get("_id", ""),
                }
                
                # Include metadata if configured
                if self.config.include_metadata:
                    item["metadata"] = doc.get("metadata", {})
                
                # Include content if configured
                if self.config.include_content:
                    item["text"] = doc.get("text", "")
                
                items.append(item)
            
            return SearchResult(
                items=items,
                total_count=len(items),  # This would ideally be the total count, not just the page count
                metadata={
                    "category": category,
                    "page": page,
                    "page_size": page_size,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            
        except Exception as e:
            return SearchResult(
                items=[],
                total_count=0,
                metadata={"category": category},
                error=str(e),
            )
    
    async def suggest(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """Suggest query completions"""
        try:
            # This is a simple implementation that just uses recent queries
            # A more sophisticated implementation would use a dedicated suggestion engine
            suggestions = []
            
            # Check recent queries
            for query in self.recent_queries:
                if query.lower().startswith(partial_query.lower()):
                    suggestions.append(query)
                    if len(suggestions) >= max_suggestions:
                        break
            
            # If we don't have enough suggestions, add some generic ones
            if len(suggestions) < max_suggestions:
                generic_suggestions = [
                    f"{partial_query} definition",
                    f"{partial_query} examples",
                    f"{partial_query} tutorial",
                    f"{partial_query} vs",
                    f"how to use {partial_query}",
                ]
                
                for suggestion in generic_suggestions:
                    if suggestion not in suggestions:
                        suggestions.append(suggestion)
                        if len(suggestions) >= max_suggestions:
                            break
            
            return suggestions
            
        except Exception as e:
            print(f"Error generating suggestions: {str(e)}")
            return []
    
    def _track_query(self, query: str) -> None:
        """Track a query for analytics"""
        # Add to recent queries
        self.recent_queries.insert(0, query)
        
        # Keep only the most recent queries
        self.recent_queries = self.recent_queries[:100]
    
    def _track_category(self, category: str) -> None:
        """Track a category browse for analytics"""
        # Increment the category count
        self.popular_categories[category] = self.popular_categories.get(category, 0) + 1