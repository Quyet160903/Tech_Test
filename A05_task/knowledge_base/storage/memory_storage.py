"""In-memory storage backend for the Knowledge Base System"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from knowledge_base.storage import BaseStorage, StorageConfig, StorageResult


class MemoryStorage(BaseStorage):
    """Simple in-memory storage backend for testing"""
    
    def __init__(self):
        self.documents = {}
        self.initialized = False
    
    async def initialize(self, config: StorageConfig) -> bool:
        """Initialize the memory storage"""
        try:
            self.initialized = True
            print(f"Initialized memory storage: {config.collection_name}")
            return True
        except Exception as e:
            print(f"Failed to initialize memory storage: {str(e)}")
            return False
    
    async def store(self, documents: List[Dict[str, Any]]) -> StorageResult:
        """Store documents in memory"""
        if not self.initialized:
            return StorageResult(
                success=False,
                error="Memory storage not initialized",
            )
        
        try:
            stored_ids = []
            timestamp = datetime.now()
            
            for doc in documents:
                # Generate ID if not present
                doc_id = doc.get("id", str(uuid.uuid4()))
                
                # Create a copy and add metadata
                doc_copy = doc.copy()
                doc_copy["id"] = doc_id
                doc_copy["stored_at"] = timestamp
                
                # Store in memory
                self.documents[doc_id] = doc_copy
                stored_ids.append(doc_id)
            
            return StorageResult(
                success=True,
                document_ids=stored_ids,
                count=len(stored_ids),
                metadata={"storage_type": "memory"},
            )
            
        except Exception as e:
            return StorageResult(
                success=False,
                error=str(e),
            )
    
    async def retrieve(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents from memory"""
        if not self.initialized:
            return []
        
        try:
            results = []
            count = 0
            
            for doc in self.documents.values():
                if count >= limit:
                    break
                
                # Simple query matching
                matches = True
                for key, value in query.items():
                    if key not in doc or doc[key] != value:
                        matches = False
                        break
                
                if matches:
                    results.append(doc)
                    count += 1
            
            return results
            
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []
    
    async def update(self, document_id: str, updates: Dict[str, Any]) -> StorageResult:
        """Update a document in memory"""
        if not self.initialized:
            return StorageResult(
                success=False,
                error="Memory storage not initialized",
            )
        
        try:
            if document_id in self.documents:
                self.documents[document_id].update(updates)
                self.documents[document_id]["updated_at"] = datetime.now()
                
                return StorageResult(
                    success=True,
                    count=1,
                    metadata={"updated": document_id},
                )
            else:
                return StorageResult(
                    success=False,
                    error=f"Document {document_id} not found",
                )
            
        except Exception as e:
            return StorageResult(
                success=False,
                error=str(e),
            )
    
    async def delete(self, query: Dict[str, Any]) -> StorageResult:
        """Delete documents from memory"""
        if not self.initialized:
            return StorageResult(
                success=False,
                error="Memory storage not initialized",
            )
        
        try:
            deleted_ids = []
            
            # Find matching documents
            to_delete = []
            for doc_id, doc in self.documents.items():
                matches = True
                for key, value in query.items():
                    if key not in doc or doc[key] != value:
                        matches = False
                        break
                
                if matches:
                    to_delete.append(doc_id)
            
            # Delete them
            for doc_id in to_delete:
                del self.documents[doc_id]
                deleted_ids.append(doc_id)
            
            return StorageResult(
                success=True,
                count=len(deleted_ids),
                metadata={"deleted_ids": deleted_ids},
            )
            
        except Exception as e:
            return StorageResult(
                success=False,
                error=str(e),
            ) 