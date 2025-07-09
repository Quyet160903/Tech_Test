"""Web extractor for the Knowledge Base System"""

import asyncio
import re
from typing import Dict, List, Any, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pydantic import HttpUrl

from knowledge_base.extractors import BaseExtractor, ExtractorConfig, ExtractorResult


class WebExtractor(BaseExtractor):
    """Extractor for web pages"""
    
    def __init__(self):
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        self.user_agent = "KnowledgeBaseBot/1.0"
        
    async def validate_source(self, source_url: str) -> bool:
        """Validate if a URL is accessible and is a web page"""
        try:
            # Try HEAD request first
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            try:
                response = self.session.head(source_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    content_type = response.headers.get("Content-Type", "")
                    if "text/html" in content_type.lower():
                        return True
            except Exception:
                pass
            
            # If HEAD fails, try GET with limited content
            try:
                response = self.session.get(source_url, headers=headers, timeout=10, stream=True)
                if response.status_code == 200:
                    content_type = response.headers.get("Content-Type", "")
                    # Accept if it looks like HTML or if we can't determine
                    return "text/html" in content_type.lower() or "text/" in content_type.lower() or not content_type
            except Exception:
                pass
            
            return False
        except Exception:
            return False
    
    async def extract(self, config: ExtractorConfig) -> ExtractorResult:
        """Extract content from a website"""
        if not config.source_id:
            return ExtractorResult(
                source_id=config.source_id,
                content=[],
                metadata={},
                status="failed",
                error="Missing source URL",
            )
        
        try:
            # Reset visited URLs for this extraction
            self.visited_urls = set()
            
            # Get the base URL from the config
            base_url = config.filters.get("source") if config.filters else None
            if not base_url:
                return ExtractorResult(
                    source_id=config.source_id,
                    content=[],
                    metadata={},
                    status="failed",
                    error="Missing URL in filters",
                )
            
            # Start the crawling process
            all_content = []
            metadata = {
                "pages_processed": 0,
                "total_content_length": 0,
                "base_url": base_url,
            }
            
            # Process the base URL first
            await self._process_url(base_url, all_content, config, metadata)
            
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
    
    async def _process_url(
        self, url: str, all_content: List[Dict[str, Any]], config: ExtractorConfig, metadata: Dict[str, Any]
    ) -> None:
        """Process a single URL and extract content"""
        if url in self.visited_urls:
            return
        
        if len(self.visited_urls) >= config.max_pages:
            return
        
        self.visited_urls.add(url)
        
        try:
            # Fetch the page
            headers = {"User-Agent": self.user_agent}
            if config.headers:
                headers.update(config.headers)
                
            response = self.session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, "lxml")
            
            # Extract the main content
            content = self._extract_content(soup, url)
            
            if content:
                all_content.append(content)
                metadata["pages_processed"] += 1
                metadata["total_content_length"] += len(content.get("text", ""))
            
            # Find links to follow if configured
            if config.follow_links and len(self.visited_urls) < config.max_pages:
                links = self._extract_links(soup, url, config.max_depth)
                
                # Process links asynchronously with rate limiting
                tasks = []
                for link in links[:config.max_pages - len(self.visited_urls)]:
                    # Add a small delay to avoid overwhelming the server
                    await asyncio.sleep(0.5)
                    tasks.append(self._process_url(link, all_content, config, metadata))
                
                await asyncio.gather(*tasks)
                
        except Exception as e:
            # Log the error but continue with other URLs
            print(f"Error processing URL {url}: {str(e)}")
    
    def _extract_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract content from a web page"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
        
        # Get the page title
        title = soup.title.string if soup.title else "Untitled"
        
        # Get the main content
        main_content = ""
        
        # Try to find the main content container
        main_elements = soup.select("main, article, .content, #content, .main, #main")
        if main_elements:
            main_content = main_elements[0].get_text(separator="\n", strip=True)
        else:
            # Fall back to the body
            main_content = soup.body.get_text(separator="\n", strip=True) if soup.body else ""
        
        # Clean up the content
        main_content = re.sub(r"\n{3,}", "\n\n", main_content)
        
        # Extract metadata
        meta_tags = {}
        for meta in soup.find_all("meta"):
            name = meta.get("name") or meta.get("property")
            content = meta.get("content")
            if name and content:
                meta_tags[name] = content
        
        return {
            "url": url,
            "title": title,
            "text": main_content,
            "html": str(soup),
            "metadata": {
                "meta_tags": meta_tags,
                "length": len(main_content),
                "timestamp": None,  # Will be filled by the content processor
            },
        }
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str, max_depth: int) -> List[str]:
        """Extract links from a web page"""
        if max_depth <= 0:
            return []
        
        links = []
        base_domain = urlparse(base_url).netloc
        
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            
            # Skip empty links, anchors, or javascript
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            
            # Only include links from the same domain
            if urlparse(absolute_url).netloc == base_domain:
                links.append(absolute_url)
        
        # Remove duplicates and already visited URLs
        return list(set(links) - self.visited_urls) 