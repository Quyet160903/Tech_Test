"""XML extractor for the Knowledge Base System"""

import asyncio
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import requests
from urllib.parse import urljoin, urlparse

from knowledge_base.extractors import BaseExtractor, ExtractorConfig, ExtractorResult


class XMLExtractor(BaseExtractor):
    """Extractor for XML documents and RSS feeds"""
    
    def __init__(self):
        self.session = requests.Session()
        self.user_agent = "KnowledgeBaseBot/1.0"
        
    async def validate_source(self, source_location: str) -> bool:
        """Validate if a source is a valid XML document"""
        try:
            if source_location.startswith(('http://', 'https://')):
                # Validate URL-based XML
                headers = {"User-Agent": self.user_agent}
                response = self.session.head(source_location, headers=headers, timeout=10)
                if response.status_code == 200:
                    content_type = response.headers.get("Content-Type", "")
                    return any(xml_type in content_type.lower() for xml_type in 
                              ["xml", "rss", "atom", "application/xml", "text/xml"])
            else:
                # Validate local XML file
                try:
                    ET.parse(source_location)
                    return True
                except ET.ParseError:
                    return False
                    
            return False
            
        except Exception:
            return False
    
    async def extract(self, config: ExtractorConfig) -> ExtractorResult:
        """Extract content from XML documents"""
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
            
            # Parse the XML document
            if source_location.startswith(('http://', 'https://')):
                # Fetch from URL
                headers = {"User-Agent": self.user_agent}
                response = self.session.get(source_location, headers=headers, timeout=30)
                response.raise_for_status()
                xml_content = response.text
                root = ET.fromstring(xml_content)
            else:
                # Parse local file
                tree = ET.parse(source_location)
                root = tree.getroot()
            
            # Detect XML type and extract accordingly
            content = []
            metadata = {
                "xml_type": self._detect_xml_type(root),
                "root_tag": root.tag,
                "total_elements": len(list(root.iter())),
                "source_url": source_location,
            }
            
            if self._is_rss_feed(root):
                content = self._extract_rss_content(root, source_location)
                metadata["xml_type"] = "rss_feed"
            elif self._is_atom_feed(root):
                content = self._extract_atom_content(root, source_location)
                metadata["xml_type"] = "atom_feed"
            else:
                content = self._extract_generic_xml_content(root, source_location)
                metadata["xml_type"] = "generic_xml"
            
            return ExtractorResult(
                source_id=config.source_id,
                content=content,
                metadata=metadata,
                status="completed" if content else "empty",
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
    
    def _detect_xml_type(self, root: ET.Element) -> str:
        """Detect the type of XML document"""
        root_tag = root.tag.lower()
        
        if "rss" in root_tag:
            return "rss"
        elif "feed" in root_tag:
            return "atom"
        elif "sitemap" in root_tag:
            return "sitemap"
        elif any(tag in root_tag for tag in ["book", "article", "paper"]):
            return "document"
        else:
            return "generic"
    
    def _is_rss_feed(self, root: ET.Element) -> bool:
        """Check if XML is an RSS feed"""
        return root.tag.lower() == "rss" or any(
            child.tag.lower() == "channel" for child in root
        )
    
    def _is_atom_feed(self, root: ET.Element) -> bool:
        """Check if XML is an Atom feed"""
        return "feed" in root.tag.lower() or root.tag.lower() == "feed"
    
    def _extract_rss_content(self, root: ET.Element, source_url: str) -> List[Dict[str, Any]]:
        """Extract content from RSS feed"""
        content = []
        
        # Find channel
        channel = root.find("channel") or root
        
        # Extract channel info
        channel_title = self._get_text(channel.find("title"))
        channel_description = self._get_text(channel.find("description"))
        
        # Extract items
        for item in channel.findall("item"):
            title = self._get_text(item.find("title"))
            description = self._get_text(item.find("description"))
            link = self._get_text(item.find("link"))
            pub_date = self._get_text(item.find("pubDate"))
            category = self._get_text(item.find("category"))
            
            # Combine title and description
            text_content = f"{title}\n\n{description}" if title and description else (title or description or "")
            
            if text_content:
                content.append({
                    "title": title or "Untitled",
                    "text": text_content,
                    "url": link or source_url,
                    "source_id": f"rss_{hash(link or title or text_content)}",
                    "metadata": {
                        "channel_title": channel_title,
                        "channel_description": channel_description,
                        "publication_date": pub_date,
                        "category": category,
                        "content_type": "rss_item",
                        "source_type": "rss_feed",
                    }
                })
        
        return content
    
    def _extract_atom_content(self, root: ET.Element, source_url: str) -> List[Dict[str, Any]]:
        """Extract content from Atom feed"""
        content = []
        
        # Extract feed info
        feed_title = self._get_text(root.find("title"))
        
        # Handle namespaces for Atom
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall(".//atom:entry", namespace) or root.findall("entry")
        
        for entry in entries:
            title = self._get_text(entry.find("title"))
            summary = self._get_text(entry.find("summary")) or self._get_text(entry.find("content"))
            
            # Find link
            link_elem = entry.find("link")
            link = link_elem.get("href") if link_elem is not None else None
            
            updated = self._get_text(entry.find("updated"))
            
            text_content = f"{title}\n\n{summary}" if title and summary else (title or summary or "")
            
            if text_content:
                content.append({
                    "title": title or "Untitled",
                    "text": text_content,
                    "url": link or source_url,
                    "source_id": f"atom_{hash(link or title or text_content)}",
                    "metadata": {
                        "feed_title": feed_title,
                        "updated": updated,
                        "content_type": "atom_entry",
                        "source_type": "atom_feed",
                    }
                })
        
        return content
    
    def _extract_generic_xml_content(self, root: ET.Element, source_url: str) -> List[Dict[str, Any]]:
        """Extract content from generic XML documents"""
        content = []
        
        # Extract text content from all elements
        all_text = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                all_text.append(elem.text.strip())
        
        if all_text:
            text_content = "\n".join(all_text)
            content.append({
                "title": f"XML Document: {root.tag}",
                "text": text_content,
                "url": source_url,
                "source_id": f"xml_{hash(text_content)}",
                "metadata": {
                    "root_element": root.tag,
                    "element_count": len(list(root.iter())),
                    "content_type": "xml_document",
                    "source_type": "xml",
                    "attributes": dict(root.attrib) if root.attrib else {},
                }
            })
        
        return content
    
    def _get_text(self, element: Optional[ET.Element]) -> Optional[str]:
        """Safely get text from XML element"""
        if element is not None and element.text:
            return element.text.strip()
        return None 