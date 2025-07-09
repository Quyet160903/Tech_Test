"""YouTube extractor for the Knowledge Base System"""

import asyncio
import re
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import requests
from urllib.parse import urlparse, parse_qs

from knowledge_base.extractors import BaseExtractor, ExtractorConfig, ExtractorResult


class YouTubeExtractor(BaseExtractor):
    """Extractor for YouTube videos"""
    
    def __init__(self):
        self.session = requests.Session()
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        
    async def validate_source(self, source_url: str) -> bool:
        """Validate if a URL is a YouTube video or playlist"""
        try:
            parsed = urlparse(source_url)
            
            # Check for YouTube domains
            if parsed.netloc not in ['www.youtube.com', 'youtube.com', 'youtu.be', 'm.youtube.com']:
                return False
            
            # Check for video or playlist patterns
            if 'youtu.be' in parsed.netloc:
                return True  # Short URLs are always videos
            
            # Check for video or playlist URLs
            return any(pattern in source_url for pattern in [
                '/watch?v=', '/playlist?list=', '/channel/', '/user/', '/c/'
            ])
            
        except Exception:
            return False
    
    async def extract(self, config: ExtractorConfig) -> ExtractorResult:
        """Extract content from YouTube videos"""
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
            
            # Detect URL type
            if '/playlist?list=' in source_url:
                return await self._extract_playlist(source_url, config)
            elif '/channel/' in source_url or '/user/' in source_url or '/c/' in source_url:
                return await self._extract_channel(source_url, config)
            else:
                return await self._extract_video(source_url, config)
                
        except Exception as e:
            return ExtractorResult(
                source_id=config.source_id,
                content=[],
                metadata={},
                status="failed",
                error=str(e),
            )
    
    async def _extract_video(self, video_url: str, config: ExtractorConfig) -> ExtractorResult:
        """Extract content from a single YouTube video"""
        try:
            video_id = self._extract_video_id(video_url)
            if not video_id:
                return ExtractorResult(
                    source_id=config.source_id,
                    content=[],
                    metadata={},
                    status="failed",
                    error="Could not extract video ID from URL",
                )
            
            # Get video metadata
            video_data = await self._get_video_metadata(video_id)
            if not video_data:
                return ExtractorResult(
                    source_id=config.source_id,
                    content=[],
                    metadata={},
                    status="failed",
                    error="Could not fetch video metadata",
                )
            
            # Try to get transcript (this is a placeholder - would need actual implementation)
            transcript = await self._get_video_transcript(video_id)
            
            # Create content
            content = []
            
            # Add video metadata as content
            description = video_data.get('description', '')
            title = video_data.get('title', 'Untitled Video')
            
            # Combine title, description, and transcript
            text_content = f"Title: {title}"
            if description:
                text_content += f"\n\nDescription: {description}"
            if transcript:
                text_content += f"\n\nTranscript: {transcript}"
            
            content.append({
                "title": title,
                "text": text_content,
                "url": video_url,
                "source_id": f"youtube_{video_id}",
                "metadata": {
                    "video_id": video_id,
                    "channel": video_data.get('channel', ''),
                    "duration": video_data.get('duration', ''),
                    "view_count": video_data.get('view_count', ''),
                    "upload_date": video_data.get('upload_date', ''),
                    "tags": video_data.get('tags', []),
                    "content_type": "youtube_video",
                    "source_type": "youtube",
                    "has_transcript": bool(transcript),
                }
            })
            
            return ExtractorResult(
                source_id=config.source_id,
                content=content,
                metadata={
                    "video_count": 1,
                    "source_type": "youtube_video",
                    "video_id": video_id,
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
    
    async def _extract_playlist(self, playlist_url: str, config: ExtractorConfig) -> ExtractorResult:
        """Extract content from a YouTube playlist"""
        try:
            playlist_id = self._extract_playlist_id(playlist_url)
            if not playlist_id:
                return ExtractorResult(
                    source_id=config.source_id,
                    content=[],
                    metadata={},
                    status="failed",
                    error="Could not extract playlist ID from URL",
                )
            
            # Get playlist metadata and video list
            playlist_data = await self._get_playlist_metadata(playlist_id)
            video_ids = playlist_data.get('video_ids', [])
            
            # Limit number of videos processed
            max_videos = min(len(video_ids), config.max_pages)
            video_ids = video_ids[:max_videos]
            
            content = []
            
            # Process each video in the playlist
            for video_id in video_ids:
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                video_result = await self._extract_video(video_url, config)
                
                if video_result.content:
                    # Add playlist context to video metadata
                    for item in video_result.content:
                        item["metadata"]["playlist_id"] = playlist_id
                        item["metadata"]["playlist_title"] = playlist_data.get('title', '')
                    content.extend(video_result.content)
            
            return ExtractorResult(
                source_id=config.source_id,
                content=content,
                metadata={
                    "video_count": len(content),
                    "source_type": "youtube_playlist",
                    "playlist_id": playlist_id,
                    "playlist_title": playlist_data.get('title', ''),
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
    
    async def _extract_channel(self, channel_url: str, config: ExtractorConfig) -> ExtractorResult:
        """Extract content from a YouTube channel (recent videos)"""
        try:
            # This is a simplified implementation
            # In a real implementation, you'd use the YouTube API
            
            # For now, return a placeholder that indicates channel extraction is not fully implemented
            return ExtractorResult(
                source_id=config.source_id,
                content=[],
                metadata={
                    "source_type": "youtube_channel",
                    "channel_url": channel_url,
                    "note": "Channel extraction requires YouTube API access",
                },
                status="failed",
                error="Channel extraction not implemented - requires YouTube API",
            )
            
        except Exception as e:
            return ExtractorResult(
                source_id=config.source_id,
                content=[],
                metadata={},
                status="failed",
                error=str(e),
            )
    
    def _extract_video_id(self, video_url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, video_url)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_playlist_id(self, playlist_url: str) -> Optional[str]:
        """Extract playlist ID from YouTube URL"""
        match = re.search(r'list=([a-zA-Z0-9_-]+)', playlist_url)
        return match.group(1) if match else None
    
    async def _get_video_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get video metadata (placeholder implementation)"""
        try:
            # This is a placeholder implementation
            # In production, you would use the YouTube Data API v3
            # For now, we'll try to scrape basic info from the page
            
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            headers = {"User-Agent": self.user_agent}
            
            response = self.session.get(video_url, headers=headers, timeout=30)
            if response.status_code != 200:
                return None
            
            html_content = response.text
            
            # Extract basic metadata using regex (very basic implementation)
            title_match = re.search(r'"title":"([^"]+)"', html_content)
            title = title_match.group(1) if title_match else "Unknown Title"
            
            # Try to extract description
            description_patterns = [
                r'"shortDescription":"([^"]+)"',
                r'"description":{"simpleText":"([^"]+)"}'
            ]
            
            description = ""
            for pattern in description_patterns:
                desc_match = re.search(pattern, html_content)
                if desc_match:
                    description = desc_match.group(1)
                    break
            
            # Extract channel name
            channel_match = re.search(r'"author":"([^"]+)"', html_content)
            channel = channel_match.group(1) if channel_match else "Unknown Channel"
            
            return {
                "title": title,
                "description": description,
                "channel": channel,
                "duration": "",  # Would need more complex extraction
                "view_count": "",  # Would need more complex extraction
                "upload_date": "",  # Would need more complex extraction
                "tags": [],  # Would need more complex extraction
            }
            
        except Exception as e:
            print(f"Error getting video metadata for {video_id}: {str(e)}")
            return None
    
    async def _get_playlist_metadata(self, playlist_id: str) -> Dict[str, Any]:
        """Get playlist metadata (placeholder implementation)"""
        try:
            # This is a placeholder implementation
            # In production, you would use the YouTube Data API v3
            
            return {
                "title": f"YouTube Playlist {playlist_id}",
                "video_ids": [],  # Would need API to get actual video IDs
            }
            
        except Exception as e:
            print(f"Error getting playlist metadata for {playlist_id}: {str(e)}")
            return {"title": "Unknown Playlist", "video_ids": []}
    
    async def _get_video_transcript(self, video_id: str) -> Optional[str]:
        """Get video transcript (placeholder implementation)"""
        try:
            # This is a placeholder implementation
            # To get actual transcripts, you would need to use:
            # 1. YouTube API with captions
            # 2. Third-party libraries like youtube-transcript-api
            # 3. Or scrape the transcript data from the page
            
            # For now, return None to indicate no transcript available
            return None
            
        except Exception as e:
            print(f"Error getting transcript for {video_id}: {str(e)}")
            return None 