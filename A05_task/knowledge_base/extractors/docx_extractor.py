"""DOCX extractor for the Knowledge Base System"""

import asyncio
import os
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import tempfile
import zipfile
import xml.etree.ElementTree as ET

from knowledge_base.extractors import BaseExtractor, ExtractorConfig, ExtractorResult


class DocxExtractor(BaseExtractor):
    """Extractor for Microsoft Word DOCX documents"""
    
    def __init__(self):
        pass
        
    async def validate_source(self, source_location: str) -> bool:
        """Validate if a source is a DOCX document"""
        try:
            # Check file extension
            if source_location.lower().endswith('.docx'):
                # If it's a file path, check if it exists
                if not source_location.startswith(('http://', 'https://')):
                    return os.path.exists(source_location)
                return True
            
            return False
            
        except Exception:
            return False
    
    async def extract(self, config: ExtractorConfig) -> ExtractorResult:
        """Extract content from DOCX documents"""
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
                    error="No content extracted from DOCX file",
                )
            
            return ExtractorResult(
                source_id=config.source_id,
                content=content,
                metadata={
                    "source_type": "docx",
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
        """Extract content from a local DOCX file"""
        try:
            # Extract text from DOCX
            text_content, metadata = self._extract_docx_content(file_path)
            
            if not text_content:
                return []
            
            # Create content item
            content = [{
                "title": metadata.get('title') or os.path.basename(file_path).replace('.docx', ''),
                "text": text_content,
                "url": f"file://{os.path.abspath(file_path)}",
                "source_id": f"docx_{hash(file_path)}",
                "metadata": {
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "content_type": "docx_document",
                    "source_type": "docx",
                    "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else None,
                    "extraction_date": datetime.now().isoformat(),
                    **metadata,
                }
            }]
            
            return content
            
        except Exception as e:
            print(f"Error extracting from DOCX file {file_path}: {str(e)}")
            return []
    
    async def _extract_from_url(self, url: str, config: ExtractorConfig) -> List[Dict[str, Any]]:
        """Extract content from a DOCX file URL"""
        try:
            import requests
            
            # Download the file
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            try:
                # Extract from temporary file
                text_content, metadata = self._extract_docx_content(temp_path)
                
                if not text_content:
                    return []
                
                # Create content item
                content = [{
                    "title": metadata.get('title') or url.split('/')[-1].replace('.docx', ''),
                    "text": text_content,
                    "url": url,
                    "source_id": f"docx_{hash(url)}",
                    "metadata": {
                        "source_url": url,
                        "content_type": "docx_document",
                        "source_type": "docx",
                        "extraction_date": datetime.now().isoformat(),
                        **metadata,
                    }
                }]
                
                return content
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
        except Exception as e:
            print(f"Error extracting from DOCX URL {url}: {str(e)}")
            return []
    
    def _extract_docx_content(self, file_path: str) -> tuple:
        """Extract text content and metadata from DOCX file"""
        try:
            text_content = ""
            metadata = {}
            
            # DOCX files are ZIP archives
            with zipfile.ZipFile(file_path, 'r') as docx_zip:
                # Extract document content
                if 'word/document.xml' in docx_zip.namelist():
                    with docx_zip.open('word/document.xml') as doc_xml:
                        doc_content = doc_xml.read()
                        text_content = self._extract_text_from_xml(doc_content)
                
                # Extract metadata from core properties
                if 'docProps/core.xml' in docx_zip.namelist():
                    with docx_zip.open('docProps/core.xml') as core_xml:
                        core_content = core_xml.read()
                        metadata.update(self._extract_core_properties(core_content))
                
                # Extract app properties if available
                if 'docProps/app.xml' in docx_zip.namelist():
                    with docx_zip.open('docProps/app.xml') as app_xml:
                        app_content = app_xml.read()
                        metadata.update(self._extract_app_properties(app_content))
            
            return text_content.strip(), metadata
            
        except Exception as e:
            print(f"Error extracting DOCX content: {str(e)}")
            return "", {}
    
    def _extract_text_from_xml(self, xml_content: bytes) -> str:
        """Extract text from Word document XML"""
        try:
            root = ET.fromstring(xml_content)
            
            # Word documents use namespaces
            namespaces = {
                'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
            }
            
            # Find all text elements
            text_elements = []
            
            # Get all paragraph text
            for paragraph in root.findall('.//w:p', namespaces):
                paragraph_text = []
                for text_elem in paragraph.findall('.//w:t', namespaces):
                    if text_elem.text:
                        paragraph_text.append(text_elem.text)
                
                if paragraph_text:
                    text_elements.append(''.join(paragraph_text))
            
            # Join paragraphs with double newlines
            return '\n\n'.join(text_elements)
            
        except Exception as e:
            print(f"Error extracting text from XML: {str(e)}")
            return ""
    
    def _extract_core_properties(self, xml_content: bytes) -> Dict[str, Any]:
        """Extract core document properties"""
        try:
            root = ET.fromstring(xml_content)
            
            # Common Dublin Core namespaces
            namespaces = {
                'dc': 'http://purl.org/dc/elements/1.1/',
                'dcterms': 'http://purl.org/dc/terms/',
                'cp': 'http://schemas.openxmlformats.org/package/2006/metadata/core-properties'
            }
            
            properties = {}
            
            # Extract common properties
            property_map = {
                'title': 'dc:title',
                'creator': 'dc:creator',
                'subject': 'dc:subject',
                'description': 'dc:description',
                'created': 'dcterms:created',
                'modified': 'dcterms:modified',
                'lastModifiedBy': 'cp:lastModifiedBy',
                'revision': 'cp:revision',
            }
            
            for prop_name, xpath in property_map.items():
                elem = root.find(xpath, namespaces)
                if elem is not None and elem.text:
                    properties[prop_name] = elem.text.strip()
            
            return properties
            
        except Exception as e:
            print(f"Error extracting core properties: {str(e)}")
            return {}
    
    def _extract_app_properties(self, xml_content: bytes) -> Dict[str, Any]:
        """Extract application-specific properties"""
        try:
            root = ET.fromstring(xml_content)
            
            # Application properties namespace
            namespaces = {
                'app': 'http://schemas.openxmlformats.org/officeDocument/2006/extended-properties'
            }
            
            properties = {}
            
            # Extract useful app properties
            property_map = {
                'application': 'app:Application',
                'appVersion': 'app:AppVersion',
                'company': 'app:Company',
                'totalTime': 'app:TotalTime',
                'pages': 'app:Pages',
                'words': 'app:Words',
                'characters': 'app:Characters',
                'charactersWithSpaces': 'app:CharactersWithSpaces',
                'paragraphs': 'app:Paragraphs',
                'lines': 'app:Lines',
            }
            
            for prop_name, xpath in property_map.items():
                elem = root.find(xpath, namespaces)
                if elem is not None and elem.text:
                    # Try to convert numeric values
                    value = elem.text.strip()
                    if prop_name in ['pages', 'words', 'characters', 'charactersWithSpaces', 'paragraphs', 'lines']:
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                    properties[prop_name] = value
            
            return properties
            
        except Exception as e:
            print(f"Error extracting app properties: {str(e)}")
            return {} 