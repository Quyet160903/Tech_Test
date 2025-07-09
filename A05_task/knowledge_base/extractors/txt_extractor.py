"""Plain text extractor for the Knowledge Base System"""

import asyncio
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile
import chardet

from knowledge_base.extractors import BaseExtractor, ExtractorConfig, ExtractorResult


class TxtExtractor(BaseExtractor):
    """Extractor for plain text files"""
    
    def __init__(self):
        pass
        
    async def validate_source(self, source_location: str) -> bool:
        """Validate if a source is a plain text file"""
        try:
            # Check file extension
            if source_location.lower().endswith(('.txt', '.text', '.log', '.md', '.rst')):
                # If it's a file path, check if it exists
                if not source_location.startswith(('http://', 'https://')):
                    return os.path.exists(source_location)
                return True
            
            return False
            
        except Exception:
            return False
    
    async def extract(self, config: ExtractorConfig) -> ExtractorResult:
        """Extract content from text files"""
        try:
            source_location = config.filters.get("source") if config.filters else None
            if not source_location:
                return ExtractorResult(
                    source_id=config.source_id,
                    content=[],
                    metadata={},
                    status="failed",
                    error="Missing source location in filters",
                )
            
            # Handle both local files and URLs
            if source_location.startswith(('http://', 'https://')):
                content = await self._extract_from_url(source_location, config)
            else:
                content = await self._extract_from_file(source_location, config)
            
            if not content:
                return ExtractorResult(
                    source_id=config.source_id,
                    content=[],
                    metadata={},
                    status="empty",
                    error="No content extracted from text file",
                )
            
            return ExtractorResult(
                source_id=config.source_id,
                content=content,
                metadata={
                    "source_type": "txt",
                    "source_location": source_location,
                    "content_items": len(content),
                },
                status="completed",
                error=None,
            )
                
        except Exception as e:
            return ExtractorResult(
                source_id=config.source_id,
                content=[],
                metadata={},
                status="failed",
                error=str(e),
            )
    
    async def _extract_from_file(self, file_path: str, config: ExtractorConfig) -> List[Dict[str, Any]]:
        """Extract content from a local text file"""
        try:
            # Detect encoding
            encoding = self._detect_encoding(file_path)
            
            # Read file content
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                text_content = f.read()
            
            if not text_content.strip():
                return []
            
            # Get file stats
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            modified_time = datetime.fromtimestamp(file_stats.st_mtime)
            
            # Create content item
            content = [{
                "title": os.path.basename(file_path).replace('.txt', '').replace('.text', ''),
                "text": text_content,
                "url": f"file://{os.path.abspath(file_path)}",
                "source_id": f"txt_{hash(file_path)}",
                "metadata": {
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "content_type": "text_document",
                    "source_type": "txt",
                    "file_size": file_size,
                    "encoding": encoding,
                    "modified_time": modified_time.isoformat(),
                    "extraction_date": datetime.now().isoformat(),
                    "line_count": len(text_content.splitlines()),
                    "character_count": len(text_content),
                    "word_count": len(text_content.split()),
                }
            }]
            
            return content
            
        except Exception as e:
            print(f"Error extracting from text file {file_path}: {str(e)}")
            return []
    
    async def _extract_from_url(self, url: str, config: ExtractorConfig) -> List[Dict[str, Any]]:
        """Extract content from a text file URL"""
        try:
            import requests
            
            # Download the file
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Try to detect encoding from response headers or content
            encoding = response.encoding or 'utf-8'
            text_content = response.text
            
            if not text_content.strip():
                return []
            
            # Create content item
            content = [{
                "title": url.split('/')[-1].replace('.txt', '').replace('.text', ''),
                "text": text_content,
                "url": url,
                "source_id": f"txt_{hash(url)}",
                "metadata": {
                    "source_url": url,
                    "content_type": "text_document",
                    "source_type": "txt",
                    "encoding": encoding,
                    "extraction_date": datetime.now().isoformat(),
                    "line_count": len(text_content.splitlines()),
                    "character_count": len(text_content),
                    "word_count": len(text_content.split()),
                    "response_headers": dict(response.headers),
                }
            }]
            
            return content
            
        except Exception as e:
            print(f"Error extracting from text URL {url}: {str(e)}")
            return []
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding"""
        try:
            # Read a sample of the file to detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
            
            # Use chardet to detect encoding
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            
            # Fallback to common encodings if detection fails
            if not encoding or result.get('confidence', 0) < 0.7:
                # Try common encodings
                common_encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
                for enc in common_encodings:
                    try:
                        with open(file_path, 'r', encoding=enc) as f:
                            f.read(1000)  # Try to read first 1KB
                        return enc
                    except UnicodeDecodeError:
                        continue
                return 'utf-8'  # Final fallback
            
            return encoding
            
        except Exception:
            return 'utf-8'  # Default fallback 