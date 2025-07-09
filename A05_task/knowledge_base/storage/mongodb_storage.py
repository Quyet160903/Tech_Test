"""MongoDB storage backend for the Knowledge Base System"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from knowledge_base.storage import BaseStorage, StorageConfig, StorageResult


class MongoDBStorage(BaseStorage):
    """MongoDB storage backend"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
    
    async def initialize(self, config: StorageConfig) -> bool:
        """Initialize the MongoDB connection"""
        try:
            # Get connection string from config or environment
            connection_string = config.connection_string
            if not connection_string:
                connection_string = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            
            # Connect to MongoDB
            self.client = MongoClient(connection_string)
            self.db = self.client[config.database_name]
            self.collection = self.db[config.collection_name]
            
            # Create indexes if specified
            if config.indexes:
                for index_spec in config.indexes:
                    keys = [(k, v) for k, v in index_spec.get("keys", {}).items()]
                    if keys:
                        self.collection.create_index(keys, **index_spec.get("options", {}))
            
            # Test connection
            self.client.admin.command("ping")
            return True
            
        except Exception as e:
            print(f"Failed to initialize MongoDB connection: {str(e)}")
            return False
    
    async def store(self, documents: List[Dict[str, Any]]) -> StorageResult:
        """Store documents in MongoDB"""
        if self.collection is None:
            return StorageResult(
                success=False,
                error="MongoDB connection not initialized",
            )
        
        try:
            # Add timestamps to documents and ensure unique IDs
            timestamp = datetime.now()
            processed_docs = []
            
            for doc in documents:
                # Create a copy to avoid modifying the original
                doc_copy = doc.copy()
                
                # Generate a unique ID if not present or empty
                doc_id = doc_copy.get("id", "").strip()
                if not doc_id:
                    doc_id = str(uuid.uuid4())
                
                doc_copy["_id"] = doc_id
                doc_copy["created_at"] = doc_copy.get("created_at", timestamp)
                doc_copy["updated_at"] = timestamp
                
                processed_docs.append(doc_copy)
            
            # Insert documents
            result = self.collection.insert_many(processed_docs)
            
            return StorageResult(
                success=True,
                document_ids=list(map(str, result.inserted_ids)),
                count=len(result.inserted_ids),
                metadata={"acknowledged": result.acknowledged},
            )
            
        except PyMongoError as e:
            return StorageResult(
                success=False,
                error=str(e),
            )
    
    async def retrieve(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents from MongoDB"""
        if self.collection is None:
            return []
        
        try:
            cursor = self.collection.find(query).limit(limit)
            return list(cursor)
            
        except PyMongoError as e:
            print(f"Error retrieving documents: {str(e)}")
            return []
    
    async def update(self, document_id: str, updates: Dict[str, Any]) -> StorageResult:
        """Update a document in MongoDB"""
        if self.collection is None:
            return StorageResult(
                success=False,
                error="MongoDB connection not initialized",
            )
        
        try:
            # Add updated timestamp
            updates["updated_at"] = datetime.now()
            
            # Update the document
            result = self.collection.update_one(
                {"_id": document_id},
                {"$set": updates},
            )
            
            return StorageResult(
                success=result.modified_count > 0,
                count=result.modified_count,
                metadata={
                    "matched_count": result.matched_count,
                    "modified_count": result.modified_count,
                },
            )
            
        except PyMongoError as e:
            return StorageResult(
                success=False,
                error=str(e),
            )
    
    async def delete(self, query: Dict[str, Any]) -> StorageResult:
        """Delete documents from MongoDB"""
        if self.collection is None:
            return StorageResult(
                success=False,
                error="MongoDB connection not initialized",
            )
        
        try:
            result = self.collection.delete_many(query)
            
            return StorageResult(
                success=True,
                count=result.deleted_count,
                metadata={"deleted_count": result.deleted_count},
            )
            
        except PyMongoError as e:
            return StorageResult(
                success=False,
                error=str(e),
            )