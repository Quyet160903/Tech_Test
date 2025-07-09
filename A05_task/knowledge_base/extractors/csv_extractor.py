"""CSV extractor for the Knowledge Base System"""

import csv
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import aiofiles
import aiohttp
from io import StringIO

from knowledge_base.extractors import BaseExtractor, ExtractorConfig, ExtractorResult
from knowledge_base.models.content import ContentItem


class CsvExtractor(BaseExtractor):
    """Extractor for CSV files and data"""
    
    def __init__(self):
        super().__init__()
        self.name = "csv_extractor"
        self.supported_sources = ["csv_file", "csv_url"]
    
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
                return os.path.exists(source) and source.endswith('.csv')
        except Exception:
            return False
    
    async def extract(self, config: ExtractorConfig) -> ExtractorResult:
        """Extract content from CSV source"""
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
                csv_content = await self._extract_from_url(source)
            else:
                csv_content = await self._extract_from_file(source)
            
            if csv_content is None:
                return ExtractorResult(
                    source_id=config.source_id,
                    content=[],
                    metadata={},
                    status="failed",
                    error="Failed to read CSV data"
                )
            
            # Parse CSV and convert to content items
            content_items = await self._process_csv_data(csv_content, config)
            
            return ExtractorResult(
                source_id=config.source_id,
                content=content_items,
                metadata={
                    "source_type": "csv",
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
    
    async def _extract_from_url(self, url: str) -> Optional[str]:
        """Extract CSV from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
            return None
        except Exception:
            return None
    
    async def _extract_from_file(self, file_path: str) -> Optional[str]:
        """Extract CSV from file"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                return await file.read()
        except Exception:
            return None
    
    async def _process_csv_data(self, csv_content: str, config: ExtractorConfig) -> List[ContentItem]:
        """Process CSV data into content items"""
        content_items = []
        
        # Configuration options
        delimiter = config.filters.get("delimiter", ",")
        text_columns = config.filters.get("text_columns", [])
        title_column = config.filters.get("title_column", "")
        combine_columns = config.filters.get("combine_columns", True)
        skip_header = config.filters.get("skip_header", True)
        max_rows = config.filters.get("max_rows", 1000)
        
        try:
            # Parse CSV
            csv_reader = csv.DictReader(StringIO(csv_content), delimiter=delimiter)
            
            row_count = 0
            for i, row in enumerate(csv_reader):
                if row_count >= max_rows:
                    break
                
                content_item = await self._create_content_item(
                    row, i, text_columns, title_column, combine_columns, config.source_id
                )
                if content_item:
                    content_items.append(content_item)
                    row_count += 1
        
        except Exception as e:
            print(f"Error processing CSV data: {str(e)}")
        
        return content_items
    
    async def _create_content_item(self, row: Dict[str, str], row_index: int,
                                 text_columns: List[str], title_column: str,
                                 combine_columns: bool, source_id: str) -> Optional[ContentItem]:
        """Create a content item from CSV row"""
        try:
            # Extract text content
            text_content = ""
            
            if text_columns:
                # Use specified columns
                for column in text_columns:
                    if column in row and row[column]:
                        text_content += f"{column}: {row[column]}\n"
            elif combine_columns:
                # Combine all columns
                for column, value in row.items():
                    if value and value.strip():
                        text_content += f"{column}: {value}\n"
            else:
                # Use first non-empty column as text
                for value in row.values():
                    if value and value.strip():
                        text_content = value
                        break
            
            if not text_content.strip():
                text_content = f"CSV Row {row_index}: " + " | ".join([f"{k}:{v}" for k, v in row.items()])
            
            # Extract title
            title = ""
            if title_column and title_column in row:
                title = row[title_column]
            else:
                # Use first non-empty value as title
                for value in row.values():
                    if value and value.strip():
                        title = value[:100]  # Limit title length
                        break
            
            if not title:
                title = f"CSV Row {row_index + 1}"
            
            # Create metadata
            metadata = {
                "row_index": row_index,
                "columns": list(row.keys()),
                "original_row": row
            }
            
            return ContentItem(
                id=f"{source_id}_row_{row_index}",
                text=text_content.strip(),
                title=title,
                url="",
                source_id=source_id,
                metadata=metadata,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error creating content item from CSV row: {str(e)}")
            return None 