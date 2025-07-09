"""JSON extractor for the Knowledge Base System"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import aiofiles
import aiohttp

from knowledge_base.extractors import BaseExtractor, ExtractorConfig, ExtractorResult
from knowledge_base.models.content import ContentItem


class JsonExtractor(BaseExtractor):
    """Extractor for JSON files and API responses"""
    
    def __init__(self):
        super().__init__()
        self.name = "json_extractor"
        self.supported_sources = ["json_file", "json_api", "json_url"]
    
    async def validate_source(self, source: str) -> bool:
        """Validate if the source is accessible"""
        try:
            if source.startswith(('http://', 'https://')):
                # Validate URL
                async with aiohttp.ClientSession() as session:
                    async with session.head(source) as response:
                        return response.status == 200
            else:
                # Validate file path
                import os
                return os.path.exists(source) and source.endswith('.json')
        except Exception:
            return False
    
    async def extract(self, config: ExtractorConfig) -> ExtractorResult:
        """Extract content from JSON source"""
        try:
            source = config.filters.get("source", "")
            if not source:
                return ExtractorResult(
                    source_id=config.source_id,
                    content=[],
                    metadata={},
                    status="failed",
                    error="No source specified in config"
                )
            
            # Extract based on source type
            if source.startswith(('http://', 'https://')):
                json_data = await self._extract_from_url(source)
            else:
                json_data = await self._extract_from_file(source)
            
            if json_data is None:
                return ExtractorResult(
                    source_id=config.source_id,
                    content=[],
                    metadata={},
                    status="failed",
                    error="Failed to parse JSON data"
                )
            
            # Convert JSON to content items
            content_items = await self._process_json_data(json_data, config)
            
            return ExtractorResult(
                source_id=config.source_id,
                content=content_items,
                metadata={
                    "source_type": "json",
                    "source": source,
                    "extracted_at": datetime.now(),
                    "total_items": len(content_items)
                },
                status="completed"
            )
            
        except Exception as e:
            return ExtractorResult(
                source_id=config.source_id,
                content=[],
                metadata={},
                status="failed",
                error=str(e)
            )
    
    async def _extract_from_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
            return None
        except Exception:
            return None
    
    async def _extract_from_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from file"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                content = await file.read()
                return json.loads(content)
        except Exception:
            return None
    
    async def _process_json_data(self, json_data: Dict[str, Any], config: ExtractorConfig) -> List[ContentItem]:
        """Process JSON data into content items"""
        content_items = []
        
        # Configuration options
        text_fields = config.filters.get("text_fields", ["content", "description", "text", "body"])
        title_fields = config.filters.get("title_fields", ["title", "name", "subject"])
        flatten_nested = config.filters.get("flatten_nested", True)
        
        try:
            if isinstance(json_data, list):
                # Handle array of objects
                for i, item in enumerate(json_data):
                    content_item = await self._create_content_item(
                        item, f"item_{i}", text_fields, title_fields, config.source_id
                    )
                    if content_item:
                        content_items.append(content_item)
            elif isinstance(json_data, dict):
                # Handle single object or nested structure
                if flatten_nested:
                    flattened_items = self._flatten_json(json_data)
                    for key, item in flattened_items.items():
                        content_item = await self._create_content_item(
                            item, key, text_fields, title_fields, config.source_id
                        )
                        if content_item:
                            content_items.append(content_item)
                else:
                    content_item = await self._create_content_item(
                        json_data, "root", text_fields, title_fields, config.source_id
                    )
                    if content_item:
                        content_items.append(content_item)
        
        except Exception as e:
            print(f"Error processing JSON data: {str(e)}")
        
        return content_items
    
    async def _create_content_item(self, data: Dict[str, Any], item_id: str, 
                                 text_fields: List[str], title_fields: List[str], 
                                 source_id: str) -> Optional[ContentItem]:
        """Create a content item from JSON data"""
        try:
            # Extract text content
            text_content = ""
            for field in text_fields:
                if field in data and isinstance(data[field], str):
                    text_content += data[field] + "\n"
            
            if not text_content.strip():
                # If no text fields found, convert the entire object to string
                text_content = json.dumps(data, indent=2)
            
            # Extract title
            title = ""
            for field in title_fields:
                if field in data and isinstance(data[field], str):
                    title = data[field]
                    break
            
            if not title:
                title = f"JSON Item {item_id}"
            
            # Create metadata
            metadata = {
                "item_id": item_id,
                "original_data": data,
                "extracted_fields": list(data.keys()) if isinstance(data, dict) else []
            }
            
            return ContentItem(
                id=f"{source_id}_{item_id}",
                text=text_content.strip(),
                title=title,
                url="",
                source_id=source_id,
                metadata=metadata,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error creating content item: {str(e)}")
            return None
    
    def _flatten_json(self, data: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
        """Flatten nested JSON structure"""
        items = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{parent_key}{sep}{key}" if parent_key else key
                
                if isinstance(value, dict):
                    items.update(self._flatten_json(value, new_key, sep))
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            items.update(self._flatten_json(item, f"{new_key}{sep}{i}", sep))
                        else:
                            items[f"{new_key}{sep}{i}"] = {"content": str(item)}
                else:
                    if new_key not in items:
                        items[new_key] = {}
                    items[new_key]["content"] = str(value)
        
        return items 