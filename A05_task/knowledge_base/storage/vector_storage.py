"""Vector storage backend for the Knowledge Base System"""

import os
from typing import Dict, List, Any, Optional, Tuple
import uuid
from datetime import datetime

import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from knowledge_base.storage import BaseStorage, StorageConfig, StorageResult


class VectorStorage(BaseStorage):
    """Vector storage backend using ChromaDB"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_function = None
    
    async def initialize(self, config: StorageConfig) -> bool:
        """Initialize the vector database"""
        try:
            # Get connection details from config or environment
            connection_string = config.connection_string
            
            # Set up embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Check if we're using Chroma Cloud or local persistent storage
            if connection_string and "chroma.cloud" in connection_string:
                # Parse Chroma Cloud connection string
                # Format: https://api-{region}.chroma.cloud/{tenant_id}/{database_id}
                parts = connection_string.split('/')
                if len(parts) < 4:
                    raise ValueError("Invalid Chroma Cloud connection string format")
                
                api_url = '/'.join(parts[:-1])  # Everything except the database ID
                database_id = parts[-1]
                
                # Get API key from environment
                api_key = os.getenv("CHROMA_API_KEY")
                if not api_key:
                    raise ValueError("CHROMA_API_KEY environment variable is required for Chroma Cloud")
                
                # Initialize Chroma Cloud client
                self.client = chromadb.HttpClient(
                    url=api_url,
                    token=api_key,
                    database=database_id
                )
                
                print(f"Connected to Chroma Cloud at {api_url}")
            else:
                # Use local persistent storage
                db_path = connection_string or os.getenv("VECTOR_DB_PATH", "./data/vector_db")
                
                # Ensure the directory exists
                os.makedirs(db_path, exist_ok=True)
                
                # Initialize ChromaDB locally
                self.client = chromadb.PersistentClient(
                    path=db_path,
                    settings=Settings(anonymized_telemetry=False)
                )
                
                print(f"Connected to local ChromaDB at {db_path}")
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=config.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Knowledge base vector storage"}
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize vector database: {str(e)}")
            return False
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata to ensure ChromaDB compatibility"""
        sanitized = {}
        
        for key, value in metadata.items():
            # Skip None values entirely
            if value is None:
                continue
                
            # Only include simple types that ChromaDB can handle
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, datetime):
                sanitized[key] = value.isoformat()
            elif hasattr(value, '__str__') and not isinstance(value, (dict, list, tuple)):
                # Convert other types to string, but skip complex objects
                sanitized[key] = str(value)
        
        return sanitized
    
    def _clean_filter(self, filter_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean and validate filter dictionary for ChromaDB"""
        if not filter_dict:
            return None
        
        cleaned = {}
        for key, value in filter_dict.items():
            # Skip None values
            if value is None:
                continue
            
            # Convert to supported types
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, datetime):
                cleaned[key] = value.isoformat()
            elif hasattr(value, '__str__') and not isinstance(value, (dict, list, tuple)):
                cleaned[key] = str(value)
        
        # Return None if filter is empty after cleaning
        return cleaned if cleaned else None
    
    async def store(self, documents: List[Dict[str, Any]]) -> StorageResult:
        """Store documents in the vector database"""
        if not self.collection:
            return StorageResult(
                success=False,
                error="Vector database not initialized",
            )
        
        try:
            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []
            
            for doc in documents:
                # Generate ID if not present
                doc_id = str(doc.get("id", uuid.uuid4()))
                ids.append(doc_id)
                
                # Get the text to embed
                text = doc.get("text", "")
                texts.append(text)
                
                # Prepare metadata (ChromaDB has limitations on metadata)
                metadata = {}
                
                # Add document metadata
                doc_metadata = doc.get("metadata", {})
                if isinstance(doc_metadata, dict):
                    metadata.update(self._sanitize_metadata(doc_metadata))
                
                # Add source information with proper null handling
                source_id = doc.get("source_id", "")
                if source_id:
                    metadata["source_id"] = str(source_id)
                
                url = doc.get("url", "")
                if url:
                    metadata["url"] = str(url)
                
                title = doc.get("title", "")
                if title:
                    metadata["title"] = str(title)
                
                # Add timestamps
                metadata["created_at"] = datetime.now().isoformat()
                
                # Add chunk information if available
                if "chunk_index" in doc:
                    metadata["chunk_index"] = int(doc["chunk_index"])
                
                if "chunk_count" in doc:
                    metadata["chunk_count"] = int(doc["chunk_count"])
                
                if "length" in doc:
                    metadata["length"] = int(doc["length"])
                
                metadatas.append(metadata)
            
            # Add documents to the collection
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            return StorageResult(
                success=True,
                document_ids=ids,
                count=len(ids),
                metadata={"collection": self.collection.name},
            )
            
        except Exception as e:
            return StorageResult(
                success=False,
                error=str(e),
            )
    
    async def retrieve(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents from the vector database using semantic search"""
        if not self.collection:
            return []
        
        try:
            # Check if this is a vector search or a metadata filter
            if "text" in query:
                # Semantic search
                query_text = query["text"]
                filter_dict = self._clean_filter(query.get("filter", {}))
                
                # Prepare query parameters
                query_params = {
                    "query_texts": [query_text],
                    "n_results": limit,
                }
                
                # Only add where clause if filter is not empty
                if filter_dict:
                    query_params["where"] = filter_dict
                
                results = self.collection.query(**query_params)
                
                # Format results
                documents = []
                if results and results["documents"] and results["documents"][0]:
                    for i, doc_text in enumerate(results["documents"][0]):
                        doc_id = results["ids"][0][i] if results["ids"] and results["ids"][0] else ""
                        metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
                        distance = results["distances"][0][i] if results["distances"] and results["distances"][0] else 0
                        
                        documents.append({
                            "id": doc_id,
                            "text": doc_text,
                            "metadata": metadata,
                            "relevance_score": max(0.0, 1.0 - (distance / 2))  # Convert distance to similarity score
                        })
                
                return documents
            else:
                # Metadata-only filter
                cleaned_query = self._clean_filter(query)
                
                # Prepare get parameters
                get_params = {"limit": limit}
                
                # Only add where clause if filter is not empty
                if cleaned_query:
                    get_params["where"] = cleaned_query
                
                results = self.collection.get(**get_params)
                
                # Format results
                documents = []
                if results and results["documents"]:
                    for i, doc_text in enumerate(results["documents"]):
                        doc_id = results["ids"][i] if results["ids"] else ""
                        metadata = results["metadatas"][i] if results["metadatas"] else {}
                        
                        documents.append({
                            "id": doc_id,
                            "text": doc_text,
                            "metadata": metadata
                        })
                
                return documents
            
        except Exception as e:
            print(f"Error retrieving documents from vector database: {str(e)}")
            return []
    
    async def update(self, document_id: str, updates: Dict[str, Any]) -> StorageResult:
        """Update a document in the vector database"""
        if not self.collection:
            return StorageResult(
                success=False,
                error="Vector database not initialized",
            )
        
        try:
            # Get the existing document
            results = self.collection.get(ids=[document_id])
            
            if not results or not results["documents"]:
                return StorageResult(
                    success=False,
                    error=f"Document with ID {document_id} not found",
                )
            
            # Prepare the updated document
            text = updates.get("text", results["documents"][0])
            
            # Update metadata
            metadata = results["metadatas"][0] if results["metadatas"] else {}
            
            # Update with new metadata
            update_metadata = updates.get("metadata", {})
            if isinstance(update_metadata, dict):
                metadata.update(self._sanitize_metadata(update_metadata))
            
            metadata["updated_at"] = datetime.now().isoformat()
            
            # Update the document
            self.collection.update(
                ids=[document_id],
                documents=[text],
                metadatas=[metadata]
            )
            
            return StorageResult(
                success=True,
                document_ids=[document_id],
                count=1,
                metadata={"updated": True},
            )
            
        except Exception as e:
            return StorageResult(
                success=False,
                error=str(e),
            )
    
    async def delete(self, query: Dict[str, Any]) -> StorageResult:
        """Delete documents from the vector database"""
        if not self.collection:
            return StorageResult(
                success=False,
                error="Vector database not initialized",
            )
        
        try:
            # Check if we're deleting by IDs or by filter
            if "ids" in query:
                ids = query["ids"]
                self.collection.delete(ids=ids)
                return StorageResult(
                    success=True,
                    count=len(ids),
                    metadata={"deleted_ids": ids},
                )
            else:
                # Clean the filter
                cleaned_query = self._clean_filter(query)
                
                if not cleaned_query:
                    return StorageResult(
                        success=False,
                        error="Invalid or empty delete query",
                    )
                
                # Get matching documents first to count them
                results = self.collection.get(where=cleaned_query)
                if not results or not results["ids"]:
                    return StorageResult(
                        success=True,
                        count=0,
                        metadata={"deleted_count": 0},
                    )
                
                # Delete the documents
                self.collection.delete(where=cleaned_query)
                
                return StorageResult(
                    success=True,
                    count=len(results["ids"]),
                    metadata={"deleted_count": len(results["ids"])},
                )
            
        except Exception as e:
            return StorageResult(
                success=False,
                error=str(e),
            )