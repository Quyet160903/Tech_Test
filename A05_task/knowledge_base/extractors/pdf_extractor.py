"""PDF extractor for the Knowledge Base System"""

import os
import tempfile
from typing import Dict, List, Any, Optional
import io

import requests
from pypdf import PdfReader
from pydantic import HttpUrl

from knowledge_base.extractors import BaseExtractor, ExtractorConfig, ExtractorResult


class PdfExtractor(BaseExtractor):
    """Extractor for PDF documents"""
    
    async def validate_source(self, source_url: str) -> bool:
        """Validate if a URL points to a PDF document"""
        try:
            # Check if it's a file path
            if os.path.isfile(source_url) and source_url.lower().endswith('.pdf'):
                return True
                
            # Check if it's a URL
            headers = {"User-Agent": "KnowledgeBaseBot/1.0"}
            response = requests.head(source_url, headers=headers, timeout=10)
            content_type = response.headers.get("Content-Type", "")
            return response.status_code == 200 and (
                "application/pdf" in content_type.lower() or
                source_url.lower().endswith('.pdf')
            )
        except Exception:
            return False
    
    async def extract(self, config: ExtractorConfig) -> ExtractorResult:
        """Extract content from a PDF document"""
        if not config.source_id:
            return ExtractorResult(
                source_id=config.source_id,
                content=[],
                metadata={},
                status="failed",
                error="Missing source ID",
            )
        
        try:
            # Get the PDF URL or path from the config
            pdf_source = config.filters.get("url") if config.filters else None
            if not pdf_source:
                return ExtractorResult(
                    source_id=config.source_id,
                    content=[],
                    metadata={},
                    status="failed",
                    error="Missing PDF source in filters",
                )
            
            # Process the PDF
            if pdf_source.startswith(('http://', 'https://')):
                # Download the PDF
                headers = {"User-Agent": "KnowledgeBaseBot/1.0"}
                if config.headers:
                    headers.update(config.headers)
                    
                response = requests.get(pdf_source, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Read PDF from memory
                pdf_data = io.BytesIO(response.content)
                pdf_reader = PdfReader(pdf_data)
            else:
                # Read PDF from file
                pdf_reader = PdfReader(pdf_source)
            
            # Extract content from the PDF
            all_content = []
            metadata = {
                "pages_processed": len(pdf_reader.pages),
                "total_content_length": 0,
                "source": pdf_source,
            }
            
            # Process each page
            for i, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text:
                        page_content = {
                            "page_num": i + 1,
                            "text": text,
                            "metadata": {
                                "length": len(text),
                                "page": i + 1,
                                "total_pages": len(pdf_reader.pages),
                            }
                        }
                        all_content.append(page_content)
                        metadata["total_content_length"] += len(text)
                except Exception as e:
                    print(f"Error extracting text from page {i+1}: {str(e)}")
            
            # Extract document metadata
            doc_info = pdf_reader.metadata
            if doc_info:
                metadata["document_info"] = {
                    "title": doc_info.get("/Title", ""),
                    "author": doc_info.get("/Author", ""),
                    "subject": doc_info.get("/Subject", ""),
                    "creator": doc_info.get("/Creator", ""),
                    "producer": doc_info.get("/Producer", ""),
                }
            
            return ExtractorResult(
                source_id=config.source_id,
                content=all_content,
                metadata=metadata,
                status="completed" if all_content else "empty",
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