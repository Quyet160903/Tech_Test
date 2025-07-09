"""Unified extraction manager for handling multiple source types"""

import asyncio
from typing import List, Dict, Any, Optional, Type
from datetime import datetime
import mimetypes
import os
from urllib.parse import urlparse

from knowledge_base.extractors import BaseExtractor, ExtractorConfig, ExtractorResult
from knowledge_base.extractors.web_extractor import WebExtractor
from knowledge_base.extractors.pdf_extractor import PdfExtractor
from knowledge_base.extractors.json_extractor import JsonExtractor
from knowledge_base.extractors.csv_extractor import CsvExtractor
from knowledge_base.utils.logging import logger


class SourceType:
    """Enumeration of supported source types"""
    WEB = "web"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    TXT = "txt"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    RSS = "rss"
    API = "api"
    DATABASE = "database"
    GITHUB = "github"
    ARXIV = "arxiv"
    YOUTUBE = "youtube"


class SourceInfo:
    """Information about a data source"""
    def __init__(self, source_id: str, source_type: str, location: str, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.source_id = source_id
        self.source_type = source_type
        self.location = location
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.last_extracted = None
        self.extraction_count = 0
        self.status = "pending"  # pending, processing, completed, failed


class ExtractionManager:
    """Unified manager for handling extraction from multiple source types"""
    
    def __init__(self):
        self.extractors: Dict[str, BaseExtractor] = {}
        self.sources: Dict[str, SourceInfo] = {}
        self._register_extractors()
    
    def _register_extractors(self):
        """Register all available extractors"""
        # Register existing extractors
        self.extractors[SourceType.WEB] = WebExtractor()
        self.extractors[SourceType.PDF] = PdfExtractor()
        self.extractors[SourceType.JSON] = JsonExtractor()
        self.extractors[SourceType.CSV] = CsvExtractor()
        
        # Register new extractors
        from knowledge_base.extractors.xml_extractor import XMLExtractor
        from knowledge_base.extractors.youtube_extractor import YouTubeExtractor
        from knowledge_base.extractors.github_extractor import GitHubExtractor
        from knowledge_base.extractors.arxiv_extractor import ArxivExtractor
        from knowledge_base.extractors.docx_extractor import DocxExtractor
        from knowledge_base.extractors.txt_extractor import TxtExtractor
        from knowledge_base.extractors.api_extractor import ApiExtractor
        
        self.extractors[SourceType.XML] = XMLExtractor()
        self.extractors[SourceType.RSS] = XMLExtractor()  # RSS is handled by XML extractor
        self.extractors[SourceType.YOUTUBE] = YouTubeExtractor()
        self.extractors[SourceType.GITHUB] = GitHubExtractor()
        self.extractors[SourceType.ARXIV] = ArxivExtractor()
        self.extractors[SourceType.DOCX] = DocxExtractor()
        self.extractors[SourceType.TXT] = TxtExtractor()
        self.extractors[SourceType.API] = ApiExtractor()
        
        logger.info(f"Registered {len(self.extractors)} extractors")
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported source types"""
        return list(self.extractors.keys())
    
    def detect_source_type(self, location: str) -> Optional[str]:
        """Auto-detect source type from location/URL"""
        try:
            # Check if it's a URL
            if location.startswith(('http://', 'https://')):
                parsed = urlparse(location)
                
                # Check for specific domains/patterns
                if 'arxiv.org' in parsed.netloc:
                    return SourceType.ARXIV
                elif 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc:
                    return SourceType.YOUTUBE
                elif 'github.com' in parsed.netloc:
                    return SourceType.GITHUB
                elif location.endswith('.rss') or (location.endswith('.xml') and 'rss' in location):
                    return SourceType.RSS
                elif location.endswith('.xml'):
                    return SourceType.XML
                elif location.endswith('.json'):
                    return SourceType.JSON
                elif location.endswith('.csv'):
                    return SourceType.CSV
                elif location.endswith('.pdf'):
                    return SourceType.PDF
                elif location.endswith('.docx'):
                    return SourceType.DOCX
                elif location.endswith(('.txt', '.log', '.md', '.rst')):
                    return SourceType.TXT
                elif any(api_indicator in location for api_indicator in ['/api/', '.json', '/v1/', '/v2/', '/rest/']):
                    return SourceType.API
                else:
                    return SourceType.WEB
            
            # Check file extension for local files
            if os.path.exists(location):
                _, ext = os.path.splitext(location.lower())
                extension_map = {
                    '.pdf': SourceType.PDF,
                    '.json': SourceType.JSON,
                    '.csv': SourceType.CSV,
                    '.xml': SourceType.XML,
                    '.txt': SourceType.TXT,
                    '.text': SourceType.TXT,
                    '.log': SourceType.TXT,
                    '.md': SourceType.TXT,
                    '.rst': SourceType.TXT,
                    '.docx': SourceType.DOCX,
                    '.xlsx': SourceType.XLSX,
                    '.pptx': SourceType.PPTX,
                }
                return extension_map.get(ext)
            
            # Try to detect from MIME type
            mime_type, _ = mimetypes.guess_type(location)
            if mime_type:
                mime_map = {
                    'application/pdf': SourceType.PDF,
                    'application/json': SourceType.JSON,
                    'text/csv': SourceType.CSV,
                    'application/xml': SourceType.XML,
                    'text/xml': SourceType.XML,
                    'text/plain': SourceType.TXT,
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': SourceType.DOCX,
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': SourceType.XLSX,
                    'application/vnd.openxmlformats-officedocument.presentationml.presentation': SourceType.PPTX,
                }
                return mime_map.get(mime_type)
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting source type for {location}: {str(e)}")
            return None
    
    async def add_source(self, source_id: str, location: str, 
                        source_type: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a new source to be extracted"""
        try:
            # Auto-detect source type if not provided
            if not source_type:
                source_type = self.detect_source_type(location)
                if not source_type:
                    logger.error(f"Could not detect source type for: {location}")
                    return False
            
            # Check if extractor is available
            if source_type not in self.extractors:
                logger.error(f"No extractor available for source type: {source_type}")
                return False
            
            # Validate source
            extractor = self.extractors[source_type]
            if not await extractor.validate_source(location):
                logger.error(f"Source validation failed for: {location}")
                return False
            
            # Create source info
            source_info = SourceInfo(
                source_id=source_id,
                source_type=source_type,
                location=location,
                metadata=metadata
            )
            
            self.sources[source_id] = source_info
            logger.info(f"Added source: {source_id} ({source_type}) - {location}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding source {source_id}: {str(e)}")
            return False
    
    async def extract_source(self, source_id: str, 
                           extraction_config: Optional[Dict[str, Any]] = None) -> ExtractorResult:
        """Extract content from a specific source"""
        try:
            if source_id not in self.sources:
                return ExtractorResult(
                    source_id=source_id,
                    content=[],
                    metadata={},
                    status="failed",
                    error=f"Source not found: {source_id}"
                )
            
            source_info = self.sources[source_id]
            source_info.status = "processing"
            
            # Get the appropriate extractor
            extractor = self.extractors[source_info.source_type]
            
            # Prepare extraction config
            config = ExtractorConfig(
                source_id=source_id,
                filters={
                    "source": source_info.location,
                    **(extraction_config or {}),
                    **(source_info.metadata or {})
                }
            )
            
            # Perform extraction
            logger.info(f"Starting extraction for source: {source_id}")
            result = await extractor.extract(config)
            
            # Update source info
            source_info.last_extracted = datetime.now()
            source_info.extraction_count += 1
            
            if result.error:
                source_info.status = "failed"
                logger.error(f"Extraction failed for {source_id}: {result.error}")
            else:
                source_info.status = "completed"
                logger.info(f"Extraction completed for {source_id}: {len(result.content)} items")
            
            return result
            
        except Exception as e:
            if source_id in self.sources:
                self.sources[source_id].status = "failed"
            
            logger.error(f"Error extracting source {source_id}: {str(e)}")
            return ExtractorResult(
                source_id=source_id,
                content=[],
                metadata={},
                status="failed",
                error=str(e)
            )
    
    async def extract_all_sources(self, max_concurrent: int = 3) -> Dict[str, ExtractorResult]:
        """Extract content from all sources with concurrency control"""
        results = {}
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_with_semaphore(source_id: str):
            async with semaphore:
                return await self.extract_source(source_id)
        
        # Create tasks for all sources
        tasks = []
        for source_id in self.sources:
            if self.sources[source_id].status in ["pending", "failed"]:
                task = asyncio.create_task(
                    extract_with_semaphore(source_id),
                    name=f"extract_{source_id}"
                )
                tasks.append((source_id, task))
        
        # Execute all tasks
        logger.info(f"Starting extraction for {len(tasks)} sources")
        for source_id, task in tasks:
            try:
                result = await task
                results[source_id] = result
            except Exception as e:
                logger.error(f"Error in extraction task for {source_id}: {str(e)}")
                results[source_id] = ExtractorResult(
                    source_id=source_id,
                    content=[],
                    metadata={},
                    status="failed",
                    error=str(e)
                )
        
        return results
    
    def get_source_info(self, source_id: str) -> Optional[SourceInfo]:
        """Get information about a specific source"""
        return self.sources.get(source_id)
    
    def get_all_sources(self) -> Dict[str, SourceInfo]:
        """Get information about all sources"""
        return self.sources.copy()
    
    def remove_source(self, source_id: str) -> bool:
        """Remove a source from the manager"""
        if source_id in self.sources:
            del self.sources[source_id]
            logger.info(f"Removed source: {source_id}")
            return True
        return False
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        total_sources = len(self.sources)
        by_status = {}
        by_type = {}
        
        for source in self.sources.values():
            # Count by status
            by_status[source.status] = by_status.get(source.status, 0) + 1
            
            # Count by type
            by_type[source.source_type] = by_type.get(source.source_type, 0) + 1
        
        return {
            "total_sources": total_sources,
            "by_status": by_status,
            "by_type": by_type,
            "supported_types": self.get_supported_types()
        } 