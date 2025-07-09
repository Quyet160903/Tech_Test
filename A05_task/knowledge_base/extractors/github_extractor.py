"""GitHub extractor for the Knowledge Base System"""

import asyncio
import re
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import requests
from urllib.parse import urlparse, urljoin
import base64

from knowledge_base.extractors import BaseExtractor, ExtractorConfig, ExtractorResult


class GitHubExtractor(BaseExtractor):
    """Extractor for GitHub repositories and content"""
    
    def __init__(self):
        self.session = requests.Session()
        self.api_base = "https://api.github.com"
        self.user_agent = "KnowledgeBaseBot/1.0"
        
        # Set up authentication if available
        import os
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            self.session.headers.update({
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json"
            })
        
        self.session.headers.update({"User-Agent": self.user_agent})
        
    async def validate_source(self, source_url: str) -> bool:
        """Validate if a URL is a GitHub repository or resource"""
        try:
            parsed = urlparse(source_url)
            
            # Check for GitHub domain
            if parsed.netloc not in ['github.com', 'www.github.com']:
                return False
            
            # Check for valid GitHub URL patterns
            path_patterns = [
                r'^/[^/]+/[^/]+/?$',  # Repository root
                r'^/[^/]+/[^/]+/tree/',  # Repository tree/branch
                r'^/[^/]+/[^/]+/blob/',  # File blob
                r'^/[^/]+/[^/]+/issues',  # Issues
                r'^/[^/]+/[^/]+/wiki',  # Wiki
                r'^/[^/]+/[^/]+/releases',  # Releases
            ]
            
            return any(re.match(pattern, parsed.path) for pattern in path_patterns)
            
        except Exception:
            return False
    
    async def extract(self, config: ExtractorConfig) -> ExtractorResult:
        """Extract content from GitHub repositories"""
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
            
            # Parse the GitHub URL
            owner, repo, resource_type, resource_path = self._parse_github_url(source_url)
            if not owner or not repo:
                return ExtractorResult(
                    source_id=config.source_id,
                    content=[],
                    metadata={},
                    status="failed",
                    error="Could not parse GitHub repository from URL",
                )
            
            # Extract based on resource type
            if resource_type == "issues":
                return await self._extract_issues(owner, repo, config)
            elif resource_type == "wiki":
                return await self._extract_wiki(owner, repo, config)
            elif resource_type == "blob":
                return await self._extract_file(owner, repo, resource_path, config)
            else:
                # Default: extract repository overview and documentation
                return await self._extract_repository(owner, repo, config)
                
        except Exception as e:
            return ExtractorResult(
                source_id=config.source_id,
                content=[],
                metadata={},
                status="failed",
                error=str(e),
            )
    
    async def _extract_repository(self, owner: str, repo: str, config: ExtractorConfig) -> ExtractorResult:
        """Extract repository overview and documentation"""
        try:
            content = []
            
            # Get repository metadata
            repo_data = await self._get_repository_info(owner, repo)
            if not repo_data:
                return ExtractorResult(
                    source_id=config.source_id,
                    content=[],
                    metadata={},
                    status="failed",
                    error="Could not fetch repository information",
                )
            
            # Add repository description as content
            description = repo_data.get('description', '')
            if description:
                content.append({
                    "title": f"{owner}/{repo} - Repository Description",
                    "text": description,
                    "url": f"https://github.com/{owner}/{repo}",
                    "source_id": f"github_{owner}_{repo}_description",
                    "metadata": {
                        "repository": f"{owner}/{repo}",
                        "content_type": "repository_description",
                        "source_type": "github",
                        "language": repo_data.get('language', ''),
                        "stars": repo_data.get('stargazers_count', 0),
                        "forks": repo_data.get('forks_count', 0),
                        "topics": repo_data.get('topics', []),
                    }
                })
            
            # Get and extract README
            readme_content = await self._get_readme(owner, repo)
            if readme_content:
                content.append({
                    "title": f"{owner}/{repo} - README",
                    "text": readme_content,
                    "url": f"https://github.com/{owner}/{repo}#readme",
                    "source_id": f"github_{owner}_{repo}_readme",
                    "metadata": {
                        "repository": f"{owner}/{repo}",
                        "content_type": "readme",
                        "source_type": "github",
                        "language": repo_data.get('language', ''),
                    }
                })
            
            # Extract documentation files if requested
            if config.max_pages > 1:
                doc_files = await self._get_documentation_files(owner, repo)
                for doc_file in doc_files[:config.max_pages - len(content)]:
                    file_content = await self._get_file_content(owner, repo, doc_file['path'])
                    if file_content:
                        content.append({
                            "title": f"{owner}/{repo} - {doc_file['name']}",
                            "text": file_content,
                            "url": f"https://github.com/{owner}/{repo}/blob/main/{doc_file['path']}",
                            "source_id": f"github_{owner}_{repo}_doc_{doc_file['name']}",
                            "metadata": {
                                "repository": f"{owner}/{repo}",
                                "content_type": "documentation",
                                "source_type": "github",
                                "file_path": doc_file['path'],
                                "file_type": doc_file.get('type', 'file'),
                            }
                        })
            
            return ExtractorResult(
                source_id=config.source_id,
                content=content,
                metadata={
                    "repository": f"{owner}/{repo}",
                    "source_type": "github_repository",
                    "content_count": len(content),
                    "repository_info": repo_data,
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
    
    async def _extract_issues(self, owner: str, repo: str, config: ExtractorConfig) -> ExtractorResult:
        """Extract issues from a GitHub repository"""
        try:
            issues = await self._get_issues(owner, repo, limit=config.max_pages)
            content = []
            
            for issue in issues:
                title = issue.get('title', '')
                body = issue.get('body', '')
                issue_number = issue.get('number', '')
                
                text_content = f"Issue #{issue_number}: {title}"
                if body:
                    text_content += f"\n\n{body}"
                
                content.append({
                    "title": f"{owner}/{repo} - Issue #{issue_number}: {title}",
                    "text": text_content,
                    "url": issue.get('html_url', ''),
                    "source_id": f"github_{owner}_{repo}_issue_{issue_number}",
                    "metadata": {
                        "repository": f"{owner}/{repo}",
                        "content_type": "github_issue",
                        "source_type": "github",
                        "issue_number": issue_number,
                        "state": issue.get('state', ''),
                        "labels": [label['name'] for label in issue.get('labels', [])],
                        "created_at": issue.get('created_at', ''),
                        "updated_at": issue.get('updated_at', ''),
                    }
                })
            
            return ExtractorResult(
                source_id=config.source_id,
                content=content,
                metadata={
                    "repository": f"{owner}/{repo}",
                    "source_type": "github_issues",
                    "issue_count": len(content),
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
    
    async def _extract_wiki(self, owner: str, repo: str, config: ExtractorConfig) -> ExtractorResult:
        """Extract wiki content from a GitHub repository"""
        try:
            # GitHub wikis are actually separate repositories
            # This is a placeholder implementation
            return ExtractorResult(
                source_id=config.source_id,
                content=[],
                metadata={
                    "repository": f"{owner}/{repo}",
                    "source_type": "github_wiki",
                    "note": "Wiki extraction not fully implemented",
                },
                status="failed",
                error="Wiki extraction not implemented",
            )
            
        except Exception as e:
            return ExtractorResult(
                source_id=config.source_id,
                content=[],
                metadata={},
                status="failed",
                error=str(e),
            )
    
    async def _extract_file(self, owner: str, repo: str, file_path: str, config: ExtractorConfig) -> ExtractorResult:
        """Extract content from a specific file"""
        try:
            file_content = await self._get_file_content(owner, repo, file_path)
            if not file_content:
                return ExtractorResult(
                    source_id=config.source_id,
                    content=[],
                    metadata={},
                    status="failed",
                    error="Could not fetch file content",
                )
            
            content = [{
                "title": f"{owner}/{repo} - {file_path}",
                "text": file_content,
                "url": f"https://github.com/{owner}/{repo}/blob/main/{file_path}",
                "source_id": f"github_{owner}_{repo}_file_{file_path.replace('/', '_')}",
                "metadata": {
                    "repository": f"{owner}/{repo}",
                    "content_type": "github_file",
                    "source_type": "github",
                    "file_path": file_path,
                }
            }]
            
            return ExtractorResult(
                source_id=config.source_id,
                content=content,
                metadata={
                    "repository": f"{owner}/{repo}",
                    "source_type": "github_file",
                    "file_path": file_path,
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
    
    def _parse_github_url(self, url: str) -> tuple:
        """Parse GitHub URL to extract owner, repo, resource type, and path"""
        try:
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split('/') if p]
            
            if len(path_parts) < 2:
                return None, None, None, None
            
            owner = path_parts[0]
            repo = path_parts[1]
            
            if len(path_parts) == 2:
                return owner, repo, "repository", None
            elif len(path_parts) > 2:
                resource_type = path_parts[2]
                resource_path = '/'.join(path_parts[3:]) if len(path_parts) > 3 else None
                return owner, repo, resource_type, resource_path
            
            return owner, repo, "repository", None
            
        except Exception:
            return None, None, None, None
    
    async def _get_repository_info(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """Get repository information from GitHub API"""
        try:
            url = f"{self.api_base}/repos/{owner}/{repo}"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            
            return None
            
        except Exception as e:
            print(f"Error getting repository info for {owner}/{repo}: {str(e)}")
            return None
    
    async def _get_readme(self, owner: str, repo: str) -> Optional[str]:
        """Get README content from repository"""
        try:
            url = f"{self.api_base}/repos/{owner}/{repo}/readme"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                readme_data = response.json()
                # Decode base64 content
                content = base64.b64decode(readme_data['content']).decode('utf-8')
                return content
            
            return None
            
        except Exception as e:
            print(f"Error getting README for {owner}/{repo}: {str(e)}")
            return None
    
    async def _get_file_content(self, owner: str, repo: str, file_path: str) -> Optional[str]:
        """Get content of a specific file"""
        try:
            url = f"{self.api_base}/repos/{owner}/{repo}/contents/{file_path}"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                file_data = response.json()
                if file_data.get('type') == 'file' and 'content' in file_data:
                    # Decode base64 content
                    content = base64.b64decode(file_data['content']).decode('utf-8')
                    return content
            
            return None
            
        except Exception as e:
            print(f"Error getting file content for {owner}/{repo}/{file_path}: {str(e)}")
            return None
    
    async def _get_documentation_files(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """Get list of documentation files in repository"""
        try:
            # Common documentation directories and files
            doc_paths = ['docs', 'doc', 'documentation', '.']
            doc_files = []
            
            for path in doc_paths:
                url = f"{self.api_base}/repos/{owner}/{repo}/contents/{path}" if path != '.' else f"{self.api_base}/repos/{owner}/{repo}/contents"
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    contents = response.json()
                    if isinstance(contents, list):
                        for item in contents:
                            name = item.get('name', '').lower()
                            if (item.get('type') == 'file' and 
                                any(ext in name for ext in ['.md', '.txt', '.rst', '.adoc']) and
                                any(doc_word in name for doc_word in ['readme', 'doc', 'guide', 'tutorial', 'changelog', 'contributing'])):
                                doc_files.append({
                                    'name': item.get('name', ''),
                                    'path': item.get('path', ''),
                                    'type': item.get('type', 'file'),
                                })
            
            return doc_files
            
        except Exception as e:
            print(f"Error getting documentation files for {owner}/{repo}: {str(e)}")
            return []
    
    async def _get_issues(self, owner: str, repo: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get repository issues"""
        try:
            url = f"{self.api_base}/repos/{owner}/{repo}/issues"
            params = {
                'state': 'all',
                'sort': 'updated',
                'per_page': min(limit, 100),
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            
            return []
            
        except Exception as e:
            print(f"Error getting issues for {owner}/{repo}: {str(e)}")
            return [] 