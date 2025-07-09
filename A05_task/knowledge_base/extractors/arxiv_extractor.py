"""arXiv extractor for the Knowledge Base System"""

import asyncio
import re
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import requests
from urllib.parse import urlparse, parse_qs
import xml.etree.ElementTree as ET

from knowledge_base.extractors import BaseExtractor, ExtractorConfig, ExtractorResult


class ArxivExtractor(BaseExtractor):
    """Extractor for arXiv research papers"""
    
    def __init__(self):
        self.session = requests.Session()
        self.api_base = "http://export.arxiv.org/api/query"
        self.user_agent = "KnowledgeBaseBot/1.0"
        self.session.headers.update({"User-Agent": self.user_agent})
        
    async def validate_source(self, source_url: str) -> bool:
        """Validate if a URL is an arXiv paper or search"""
        try:
            parsed = urlparse(source_url)
            
            # Check for arXiv domain
            if 'arxiv.org' not in parsed.netloc:
                return False
            
            # Check for valid arXiv URL patterns
            return any(pattern in source_url for pattern in [
                '/abs/', '/pdf/', '/list/', '/search/'
            ])
            
        except Exception:
            return False
    
    async def extract(self, config: ExtractorConfig) -> ExtractorResult:
        """Extract content from arXiv papers"""
        try:
            source_url = config.filters.get("source") if config.filters else None
            if not source_url:
                return ExtractorResult(
                    source_id=config.source_id,
                    content=[],
                    metadata={},
                    status="failed",
                    error="Missing source URL in filters",
                )
            
            # Parse the arXiv URL
            arxiv_ids = self._extract_arxiv_ids(source_url)
            if not arxiv_ids:
                return ExtractorResult(
                    source_id=config.source_id,
                    content=[],
                    metadata={},
                    status="failed",
                    error="Could not extract arXiv ID from URL",
                )
            
            # Limit the number of papers to process
            arxiv_ids = arxiv_ids[:config.max_pages]
            
            # Extract papers
            papers = await self._get_papers_metadata(arxiv_ids)
            content = []
            
            for paper in papers:
                if paper:
                    content.append(self._format_paper_content(paper))
            
            return ExtractorResult(
                source_id=config.source_id,
                content=content,
                metadata={
                    "source_type": "arxiv",
                    "paper_count": len(content),
                    "arxiv_ids": arxiv_ids,
                },
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
    
    def _extract_arxiv_ids(self, url: str) -> List[str]:
        """Extract arXiv paper IDs from URL"""
        arxiv_ids = []
        
        # Extract ID from single paper URL
        # Patterns: arxiv.org/abs/1234.5678, arxiv.org/pdf/1234.5678.pdf
        single_paper_match = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})', url)
        if single_paper_match:
            return [single_paper_match.group(1)]
        
        # For search or list URLs, we'll use the API to get recent papers
        # This is a simplified implementation
        if '/list/' in url or '/search/' in url:
            # Extract category from list URL
            category_match = re.search(r'/list/([a-z-]+)', url)
            if category_match:
                category = category_match.group(1)
                # Return placeholder for category search
                return [f"category:{category}"]
        
        return arxiv_ids
    
    async def _get_papers_metadata(self, arxiv_ids: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Get paper metadata from arXiv API"""
        papers = []
        
        for arxiv_id in arxiv_ids:
            try:
                if arxiv_id.startswith("category:"):
                    # Handle category search
                    category = arxiv_id.replace("category:", "")
                    category_papers = await self._search_by_category(category, max_results=5)
                    papers.extend(category_papers)
                else:
                    # Handle single paper
                    paper = await self._get_single_paper(arxiv_id)
                    if paper:
                        papers.append(paper)
                        
            except Exception as e:
                print(f"Error getting paper {arxiv_id}: {str(e)}")
                papers.append(None)
        
        return papers
    
    async def _get_single_paper(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a single arXiv paper"""
        try:
            params = {
                'id_list': arxiv_id,
                'max_results': 1,
            }
            
            response = self.session.get(self.api_base, params=params, timeout=30)
            if response.status_code != 200:
                return None
            
            # Parse XML response
            root = ET.fromstring(response.text)
            
            # Find the entry
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            entry = root.find('.//atom:entry', namespace)
            
            if entry is None:
                return None
            
            return self._parse_paper_entry(entry, namespace)
            
        except Exception as e:
            print(f"Error getting single paper {arxiv_id}: {str(e)}")
            return None
    
    async def _search_by_category(self, category: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search papers by category"""
        try:
            params = {
                'search_query': f'cat:{category}',
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending',
            }
            
            response = self.session.get(self.api_base, params=params, timeout=30)
            if response.status_code != 200:
                return []
            
            # Parse XML response
            root = ET.fromstring(response.text)
            
            # Find all entries
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('.//atom:entry', namespace)
            
            papers = []
            for entry in entries:
                paper = self._parse_paper_entry(entry, namespace)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Error searching category {category}: {str(e)}")
            return []
    
    def _parse_paper_entry(self, entry: ET.Element, namespace: dict) -> Optional[Dict[str, Any]]:
        """Parse a paper entry from arXiv API response"""
        try:
            # Extract basic information
            title = self._get_element_text(entry.find('atom:title', namespace))
            summary = self._get_element_text(entry.find('atom:summary', namespace))
            
            # Extract arXiv ID from the entry ID
            entry_id = self._get_element_text(entry.find('atom:id', namespace))
            arxiv_id = entry_id.split('/')[-1] if entry_id else ""
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', namespace):
                name_elem = author.find('atom:name', namespace)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())
            
            # Extract categories
            categories = []
            for category in entry.findall('atom:category', namespace):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # Extract publication date
            published = self._get_element_text(entry.find('atom:published', namespace))
            updated = self._get_element_text(entry.find('atom:updated', namespace))
            
            # Extract links
            pdf_url = ""
            abs_url = ""
            for link in entry.findall('atom:link', namespace):
                href = link.get('href', '')
                title = link.get('title', '')
                if 'pdf' in title.lower() or href.endswith('.pdf'):
                    pdf_url = href
                elif 'abs' in href:
                    abs_url = href
            
            return {
                'arxiv_id': arxiv_id,
                'title': title,
                'summary': summary,
                'authors': authors,
                'categories': categories,
                'published': published,
                'updated': updated,
                'pdf_url': pdf_url,
                'abs_url': abs_url,
            }
            
        except Exception as e:
            print(f"Error parsing paper entry: {str(e)}")
            return None
    
    def _format_paper_content(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Format paper metadata into content format"""
        authors_str = ", ".join(paper.get('authors', []))
        categories_str = ", ".join(paper.get('categories', []))
        
        # Create the main text content
        text_content = f"Title: {paper.get('title', '')}"
        
        if authors_str:
            text_content += f"\n\nAuthors: {authors_str}"
        
        if categories_str:
            text_content += f"\n\nCategories: {categories_str}"
        
        if paper.get('published'):
            text_content += f"\n\nPublished: {paper.get('published')}"
        
        if paper.get('summary'):
            text_content += f"\n\nAbstract:\n{paper.get('summary')}"
        
        return {
            "title": paper.get('title', 'Untitled Paper'),
            "text": text_content,
            "url": paper.get('abs_url', ''),
            "source_id": f"arxiv_{paper.get('arxiv_id', '')}",
            "metadata": {
                "arxiv_id": paper.get('arxiv_id', ''),
                "authors": paper.get('authors', []),
                "categories": paper.get('categories', []),
                "published": paper.get('published', ''),
                "updated": paper.get('updated', ''),
                "pdf_url": paper.get('pdf_url', ''),
                "content_type": "research_paper",
                "source_type": "arxiv",
            }
        }
    
    def _get_element_text(self, element: Optional[ET.Element]) -> str:
        """Safely get text from XML element"""
        if element is not None and element.text:
            return element.text.strip()
        return "" 