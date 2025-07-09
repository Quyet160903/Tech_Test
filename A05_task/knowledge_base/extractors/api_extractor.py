"""API extractor for the Knowledge Base System"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests
from urllib.parse import urlparse, parse_qs

from knowledge_base.extractors import BaseExtractor, ExtractorConfig, ExtractorResult


class ApiExtractor(BaseExtractor):
    """Extractor for REST APIs and web services"""
    
    def __init__(self):
        self.session = requests.Session()
        self.user_agent = "KnowledgeBaseBot/1.0"
        self.session.headers.update({"User-Agent": self.user_agent})
        
    async def validate_source(self, source_url: str) -> bool:
        """Validate if a URL is an API endpoint"""
        try:
            parsed = urlparse(source_url)
            
            # Basic URL validation
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Check for API indicators in URL
            api_indicators = [
                '/api/', '/v1/', '/v2/', '/v3/', '/rest/',
                '.json', '.xml', '/graphql', '/rpc'
            ]
            
            if any(indicator in source_url.lower() for indicator in api_indicators):
                return True
            
            # Try a HEAD request to check if it's accessible
            try:
                response = self.session.head(source_url, timeout=10)
                content_type = response.headers.get('Content-Type', '').lower()
                
                # Check for API-like content types
                if any(api_type in content_type for api_type in 
                      ['application/json', 'application/xml', 'text/xml', 'application/hal+json']):
                    return True
                    
            except Exception:
                pass
            
            return False
            
        except Exception:
            return False
    
    async def extract(self, config: ExtractorConfig) -> ExtractorResult:
        """Extract content from API endpoints"""
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
            
            # Extract API configuration from filters
            method = config.filters.get("method", "GET").upper()
            headers = config.filters.get("headers", {})
            params = config.filters.get("params", {})
            data = config.filters.get("data", {})
            auth = config.filters.get("auth", {})
            
            # Make the API request
            content = await self._make_api_request(
                source_url, method, headers, params, data, auth, config
            )
            
            if not content:
                return ExtractorResult(
                    source_id=config.source_id,
                    content=[],
                    metadata={},
                    status="empty",
                    error="No content extracted from API",
                )
            
            return ExtractorResult(
                source_id=config.source_id,
                content=content,
                metadata={
                    "source_type": "api",
                    "source_url": source_url,
                    "method": method,
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
    
    async def _make_api_request(
        self, 
        url: str, 
        method: str, 
        headers: Dict[str, str], 
        params: Dict[str, Any], 
        data: Dict[str, Any],
        auth: Dict[str, str],
        config: ExtractorConfig
    ) -> List[Dict[str, Any]]:
        """Make API request and extract content"""
        try:
            # Set up authentication
            auth_obj = None
            if auth:
                if auth.get("type") == "basic":
                    auth_obj = (auth.get("username", ""), auth.get("password", ""))
                elif auth.get("type") == "bearer" and auth.get("token"):
                    headers["Authorization"] = f"Bearer {auth['token']}"
                elif auth.get("type") == "api_key":
                    key_name = auth.get("key_name", "api_key")
                    key_value = auth.get("key_value", "")
                    if auth.get("location") == "header":
                        headers[key_name] = key_value
                    else:
                        params[key_name] = key_value
            
            # Make the request
            if method == "GET":
                response = self.session.get(url, headers=headers, params=params, auth=auth_obj, timeout=30)
            elif method == "POST":
                response = self.session.post(url, headers=headers, params=params, json=data, auth=auth_obj, timeout=30)
            elif method == "PUT":
                response = self.session.put(url, headers=headers, params=params, json=data, auth=auth_obj, timeout=30)
            elif method == "DELETE":
                response = self.session.delete(url, headers=headers, params=params, auth=auth_obj, timeout=30)
            else:
                return []
            
            response.raise_for_status()
            
            # Parse response based on content type
            content_type = response.headers.get('Content-Type', '').lower()
            
            if 'application/json' in content_type:
                return self._parse_json_response(response, url, config)
            elif 'application/xml' in content_type or 'text/xml' in content_type:
                return self._parse_xml_response(response, url, config)
            elif 'text/plain' in content_type:
                return self._parse_text_response(response, url, config)
            else:
                # Try to parse as JSON first, then fall back to text
                try:
                    return self._parse_json_response(response, url, config)
                except:
                    return self._parse_text_response(response, url, config)
            
        except Exception as e:
            print(f"Error making API request to {url}: {str(e)}")
            return []
    
    def _parse_json_response(self, response: requests.Response, url: str, config: ExtractorConfig) -> List[Dict[str, Any]]:
        """Parse JSON API response"""
        try:
            data = response.json()
            content = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of items
                for i, item in enumerate(data[:config.max_pages]):
                    content.append(self._format_json_item(item, url, f"item_{i}"))
            elif isinstance(data, dict):
                # Single object or object with nested data
                if "data" in data and isinstance(data["data"], list):
                    # Common pattern: {"data": [...], "meta": {...}}
                    for i, item in enumerate(data["data"][:config.max_pages]):
                        content.append(self._format_json_item(item, url, f"data_item_{i}"))
                elif "items" in data and isinstance(data["items"], list):
                    # Common pattern: {"items": [...]}
                    for i, item in enumerate(data["items"][:config.max_pages]):
                        content.append(self._format_json_item(item, url, f"items_{i}"))
                elif "results" in data and isinstance(data["results"], list):
                    # Common pattern: {"results": [...]}
                    for i, item in enumerate(data["results"][:config.max_pages]):
                        content.append(self._format_json_item(item, url, f"results_{i}"))
                else:
                    # Single object
                    content.append(self._format_json_item(data, url, "single_object"))
            
            return content
            
        except Exception as e:
            print(f"Error parsing JSON response: {str(e)}")
            return []
    
    def _format_json_item(self, item: Any, url: str, item_id: str) -> Dict[str, Any]:
        """Format a JSON item into content format"""
        if isinstance(item, dict):
            # Extract title from common fields
            title_fields = ["title", "name", "id", "summary", "subject", "headline"]
            title = None
            for field in title_fields:
                if field in item and item[field]:
                    title = str(item[field])
                    break
            
            # Extract text content from common fields
            text_fields = ["description", "content", "body", "text", "summary", "abstract"]
            text_content = title or "API Data"
            
            for field in text_fields:
                if field in item and item[field]:
                    if title and field != "title":
                        text_content += f"\n\n{field.title()}: {item[field]}"
                    elif not title:
                        text_content = str(item[field])
                        break
            
            # If no specific text fields found, format the entire object
            if text_content == "API Data":
                text_content = json.dumps(item, indent=2, default=str)
            
            return {
                "title": title or f"API Data - {item_id}",
                "text": text_content,
                "url": url,
                "source_id": f"api_{hash(f'{url}_{item_id}')}",
                "metadata": {
                    "source_url": url,
                    "content_type": "api_response",
                    "source_type": "api",
                    "item_id": item_id,
                    "extraction_date": datetime.now().isoformat(),
                    "raw_data": item,
                }
            }
        else:
            # Handle non-dict items (strings, numbers, etc.)
            return {
                "title": f"API Data - {item_id}",
                "text": str(item),
                "url": url,
                "source_id": f"api_{hash(f'{url}_{item_id}')}",
                "metadata": {
                    "source_url": url,
                    "content_type": "api_response",
                    "source_type": "api",
                    "item_id": item_id,
                    "extraction_date": datetime.now().isoformat(),
                    "raw_data": item,
                }
            }
    
    def _parse_xml_response(self, response: requests.Response, url: str, config: ExtractorConfig) -> List[Dict[str, Any]]:
        """Parse XML API response"""
        try:
            import xml.etree.ElementTree as ET
            
            root = ET.fromstring(response.text)
            content = []
            
            # Extract text from all elements
            all_text = []
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    all_text.append(f"{elem.tag}: {elem.text.strip()}")
            
            if all_text:
                text_content = "\n".join(all_text)
                content.append({
                    "title": f"XML API Response - {root.tag}",
                    "text": text_content,
                    "url": url,
                    "source_id": f"api_xml_{hash(url)}",
                    "metadata": {
                        "source_url": url,
                        "content_type": "api_xml_response",
                        "source_type": "api",
                        "root_element": root.tag,
                        "extraction_date": datetime.now().isoformat(),
                    }
                })
            
            return content
            
        except Exception as e:
            print(f"Error parsing XML response: {str(e)}")
            return []
    
    def _parse_text_response(self, response: requests.Response, url: str, config: ExtractorConfig) -> List[Dict[str, Any]]:
        """Parse plain text API response"""
        try:
            text_content = response.text.strip()
            
            if not text_content:
                return []
            
            content = [{
                "title": f"API Text Response",
                "text": text_content,
                "url": url,
                "source_id": f"api_text_{hash(url)}",
                "metadata": {
                    "source_url": url,
                    "content_type": "api_text_response",
                    "source_type": "api",
                    "extraction_date": datetime.now().isoformat(),
                    "response_headers": dict(response.headers),
                }
            }]
            
            return content
            
        except Exception as e:
            print(f"Error parsing text response: {str(e)}")
            return [] 