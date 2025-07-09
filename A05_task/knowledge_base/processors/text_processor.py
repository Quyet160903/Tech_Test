"""Text processor for the Knowledge Base System"""

import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from knowledge_base.processors import BaseProcessor, ProcessorConfig, ProcessorResult


class TextProcessor(BaseProcessor):
    """Processor for text content"""
    
    async def process(self, content: List[Dict[str, Any]], config: ProcessorConfig) -> ProcessorResult:
        """Process text content by cleaning and chunking"""
        if not content:
            return ProcessorResult(
                content_chunks=[],
                metadata={},
                status="empty",
                error="No content to process",
            )
        
        try:
            all_chunks = []
            metadata = {
                "original_content_count": len(content),
                "chunk_count": 0,
                "total_chunk_length": 0,
                "processing_timestamp": datetime.now().isoformat(),
            }
            
            for item in content:
                # Extract text from the content item
                text = item.get("text", "")
                if not text:
                    continue
                
                # Clean the text
                cleaned_text = self._clean_text(text)
                
                # Chunk the text
                chunks = self._chunk_text(
                    cleaned_text,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    min_chunk_size=config.min_chunk_size,
                )
                
                # Create content chunks with metadata
                for i, chunk_text in enumerate(chunks):
                    chunk = {
                        "text": chunk_text,
                        "metadata": {
                            "source_id": item.get("source_id", ""),
                            "url": item.get("url", ""),
                            "title": item.get("title", ""),
                            "page_num": item.get("page_num"),
                            "chunk_index": i,
                            "chunk_count": len(chunks),
                            "length": len(chunk_text),
                            "processed_at": datetime.now().isoformat(),
                        }
                    }
                    
                    # Copy relevant metadata from the original content
                    if "metadata" in item:
                        for key, value in item["metadata"].items():
                            if key not in chunk["metadata"]:
                                chunk["metadata"][key] = value
                    
                    all_chunks.append(chunk)
                    metadata["total_chunk_length"] += len(chunk_text)
            
            metadata["chunk_count"] = len(all_chunks)
            
            return ProcessorResult(
                content_chunks=all_chunks,
                metadata=metadata,
                status="completed" if all_chunks else "empty",
                error=None,
            )
            
        except Exception as e:
            return ProcessorResult(
                content_chunks=[],
                metadata={},
                status="failed",
                error=str(e),
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Replace multiple newlines with a single newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r" {2,}", " ", text)
        
        # Strip whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        
        return text
    
    def _chunk_text(
        self, text: str, chunk_size: int, chunk_overlap: int, min_chunk_size: int
    ) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        # If text is shorter than chunk size, return it as a single chunk
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the chunk
            end = start + chunk_size
            
            if end >= len(text):
                # Last chunk
                chunk = text[start:]
                if len(chunk) >= min_chunk_size:
                    chunks.append(chunk)
                elif chunks:
                    # Append to the previous chunk if too small
                    chunks[-1] = chunks[-1] + " " + chunk
                break
            
            # Try to find a good breaking point (end of sentence or paragraph)
            chunk_end = self._find_break_point(text, end)
            
            # Extract the chunk
            chunk = text[start:chunk_end]
            
            if len(chunk) >= min_chunk_size:
                chunks.append(chunk)
            
            # Move the start position for the next chunk
            start = chunk_end - chunk_overlap
            
            # Make sure we're making progress
            if start <= 0 or start >= len(text):
                break
        
        return chunks
    
    def _find_break_point(self, text: str, position: int) -> int:
        """Find a good breaking point near the position"""
        # Look for paragraph breaks first
        for i in range(position, max(0, position - 100), -1):
            if i < len(text) and text[i] == "\n" and i > 0 and text[i-1] == "\n":
                return i + 1
        
        # Look for sentence breaks
        for i in range(position, max(0, position - 100), -1):
            if i < len(text) and text[i] in ".!?" and (i + 1 >= len(text) or text[i+1].isspace()):
                return i + 1
        
        # Look for any whitespace
        for i in range(position, max(0, position - 50), -1):
            if i < len(text) and text[i].isspace():
                return i + 1
        
        # If no good breaking point found, just break at the position
        return position 