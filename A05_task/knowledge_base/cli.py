"""Command-line interface for the Knowledge Base System"""

import argparse
import asyncio
import sys
import os
from typing import Dict, List, Any, Optional
import json

from knowledge_base.utils.config import load_config, ensure_directories
from knowledge_base.utils.logging import logger
from knowledge_base.extractors.web_extractor import WebExtractor
from knowledge_base.extractors.pdf_extractor import PdfExtractor
from knowledge_base.processors.text_processor import TextProcessor
from knowledge_base.storage.mongodb_storage import MongoDBStorage
from knowledge_base.storage.vector_storage import VectorStorage
from knowledge_base.knowledge.entity_extractor import EntityExtractor
from knowledge_base.agent.openai_agent import OpenAIAgent
from knowledge_base.search.semantic_search import SemanticSearch


# Set up logger
# logger = setup_logger("knowledge_base.cli") # This line is removed as per the new_code


async def extract_source(source_url: str, source_type: str) -> Dict[str, Any]:
    """
    Extract content from a source
    
    Args:
        source_url: URL or path to the source
        source_type: Type of the source (website, pdf, etc.)
        
    Returns:
        Dict with extraction results
    """
    logger.info(f"Extracting content from {source_url} ({source_type})")
    
    if source_type == "website":
        extractor = WebExtractor()
        source_id = "src_" + source_url.replace("https://", "").replace("http://", "").split("/")[0].replace(".", "_")
        config = {
            "source_id": source_id,
            "filters": {"url": source_url},
            "max_pages": 10,
            "follow_links": True,
        }
    elif source_type == "pdf":
        extractor = PdfExtractor()
        source_id = "src_" + os.path.basename(source_url).split(".")[0].replace(" ", "_")
        config = {
            "source_id": source_id,
            "filters": {"url": source_url},
        }
    else:
        logger.error(f"Unsupported source type: {source_type}")
        return {"error": f"Unsupported source type: {source_type}"}
    
    # Validate the source
    if not await extractor.validate_source(source_url):
        logger.error(f"Invalid source: {source_url}")
        return {"error": f"Invalid source: {source_url}"}
    
    # Extract content
    from knowledge_base.extractors import ExtractorConfig
    result = await extractor.extract(ExtractorConfig(**config))
    
    if result.error:
        logger.error(f"Extraction error: {result.error}")
        return {"error": result.error}
    
    logger.info(f"Extracted {len(result.content)} content items")
    return {
        "source_id": source_id,
        "content": result.content,
        "metadata": result.metadata,
    }


async def process_content(content: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
    """
    Process extracted content
    
    Args:
        content: List of content items
        chunk_size: Size of content chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        Dict with processing results
    """
    logger.info(f"Processing {len(content)} content items")
    
    processor = TextProcessor()
    from knowledge_base.processors import ProcessorConfig
    result = await processor.process(content, ProcessorConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ))
    
    if result.error:
        logger.error(f"Processing error: {result.error}")
        return {"error": result.error}
    
    logger.info(f"Generated {len(result.content_chunks)} content chunks")
    return {
        "chunks": result.content_chunks,
        "metadata": result.metadata,
    }


async def store_content(chunks: List[Dict[str, Any]], db_name: str, collection_name: str) -> Dict[str, Any]:
    """
    Store processed content
    
    Args:
        chunks: List of content chunks
        db_name: Name of the database
        collection_name: Name of the collection
        
    Returns:
        Dict with storage results
    """
    logger.info(f"Storing {len(chunks)} content chunks in {db_name}.{collection_name}")
    
    # Store in MongoDB
    mongo_storage = MongoDBStorage()
    from knowledge_base.storage import StorageConfig
    if not await mongo_storage.initialize(StorageConfig(
        database_name=db_name,
        collection_name=collection_name,
    )):
        logger.error("Failed to initialize MongoDB storage")
        return {"error": "Failed to initialize MongoDB storage"}
    
    mongo_result = await mongo_storage.store(chunks)
    
    if not mongo_result.success:
        logger.error(f"MongoDB storage error: {mongo_result.error}")
        return {"error": mongo_result.error}
    
    logger.info(f"Stored {mongo_result.count} documents in MongoDB")
    
    # Store in vector database
    vector_storage = VectorStorage()
    if not await vector_storage.initialize(StorageConfig(
        database_name=db_name,
        collection_name=f"{collection_name}_vectors",
    )):
        logger.error("Failed to initialize vector storage")
        return {"error": "Failed to initialize vector storage"}
    
    vector_result = await vector_storage.store(chunks)
    
    if not vector_result.success:
        logger.error(f"Vector storage error: {vector_result.error}")
        return {"error": vector_result.error}
    
    logger.info(f"Stored {vector_result.count} documents in vector database")
    
    return {
        "mongodb_count": mongo_result.count,
        "vector_count": vector_result.count,
    }


async def extract_entities(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract entities from content chunks
    
    Args:
        chunks: List of content chunks
        
    Returns:
        Dict with extraction results
    """
    logger.info(f"Extracting entities from {len(chunks)} content chunks")
    
    extractor = EntityExtractor()
    from knowledge_base.knowledge import KnowledgeConfig
    if not await extractor.initialize(KnowledgeConfig()):
        logger.error("Failed to initialize entity extractor")
        return {"error": "Failed to initialize entity extractor"}
    
    entities = []
    relationships = []
    
    for chunk in chunks:
        # Extract entities from the chunk
        chunk_entities = await extractor.extract_entities(chunk)
        
        # Add entities
        for entity in chunk_entities:
            result = await extractor.add_entity(entity)
            if result.success:
                entities.append(entity)
        
        # Extract relationships between entities
        if len(chunk_entities) > 1:
            chunk_relationships = await extractor.extract_relationships(chunk_entities)
            
            # Add relationships
            for relationship in chunk_relationships:
                result = await extractor.add_relationship(relationship)
                if result.success:
                    relationships.append(relationship)
    
    logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
    return {
        "entities": entities,
        "relationships": relationships,
    }


async def query_knowledge_base(query: str, db_name: str, collection_name: str) -> Dict[str, Any]:
    """
    Query the knowledge base
    
    Args:
        query: Query text
        db_name: Name of the database
        collection_name: Name of the collection
        
    Returns:
        Dict with query results
    """
    logger.info(f"Querying knowledge base with: {query}")
    
    # Initialize vector storage
    vector_storage = VectorStorage()
    from knowledge_base.storage import StorageConfig
    if not await vector_storage.initialize(StorageConfig(
        database_name=db_name,
        collection_name=f"{collection_name}_vectors",
    )):
        logger.error("Failed to initialize vector storage")
        return {"error": "Failed to initialize vector storage"}
    
    # Initialize agent
    agent = OpenAIAgent(vector_storage=vector_storage)
    from knowledge_base.agent import AgentConfig
    if not await agent.initialize(AgentConfig(
        model_name=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
    )):
        logger.error("Failed to initialize AI agent")
        return {"error": "Failed to initialize AI agent"}
    
    # Query the agent
    response = await agent.query(query)
    
    if response.error:
        logger.error(f"Query error: {response.error}")
        return {"error": response.error}
    
    logger.info("Query successful")
    return {
        "answer": response.answer,
        "sources": response.sources,
        "related_entities": response.related_entities,
    }


async def search_knowledge_base(query: str, db_name: str, collection_name: str) -> Dict[str, Any]:
    """
    Search the knowledge base
    
    Args:
        query: Search query
        db_name: Name of the database
        collection_name: Name of the collection
        
    Returns:
        Dict with search results
    """
    logger.info(f"Searching knowledge base with: {query}")
    
    # Initialize vector storage
    vector_storage = VectorStorage()
    from knowledge_base.storage import StorageConfig
    if not await vector_storage.initialize(StorageConfig(
        database_name=db_name,
        collection_name=f"{collection_name}_vectors",
    )):
        logger.error("Failed to initialize vector storage")
        return {"error": "Failed to initialize vector storage"}
    
    # Initialize MongoDB storage
    mongo_storage = MongoDBStorage()
    if not await mongo_storage.initialize(StorageConfig(
        database_name=db_name,
        collection_name=collection_name,
    )):
        logger.error("Failed to initialize MongoDB storage")
        return {"error": "Failed to initialize MongoDB storage"}
    
    # Initialize search
    search = SemanticSearch(vector_storage=vector_storage, metadata_storage=mongo_storage)
    from knowledge_base.search import SearchConfig
    if not await search.initialize(SearchConfig()):
        logger.error("Failed to initialize search")
        return {"error": "Failed to initialize search"}
    
    # Search
    result = await search.search(query)
    
    if result.error:
        logger.error(f"Search error: {result.error}")
        return {"error": result.error}
    
    logger.info(f"Search found {result.total_count} results")
    return {
        "items": result.items,
        "total_count": result.total_count,
        "metadata": result.metadata,
    }


def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description="Knowledge Base System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract content from a source")
    extract_parser.add_argument("source", help="URL or path to the source")
    extract_parser.add_argument("--type", choices=["website", "pdf"], default="website", help="Type of the source")
    extract_parser.add_argument("--output", help="Output file for the extracted content")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process extracted content")
    process_parser.add_argument("input", help="Input file with extracted content")
    process_parser.add_argument("--output", help="Output file for the processed content")
    process_parser.add_argument("--chunk-size", type=int, default=1000, help="Size of content chunks")
    process_parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    
    # Store command
    store_parser = subparsers.add_parser("store", help="Store processed content")
    store_parser.add_argument("input", help="Input file with processed content")
    store_parser.add_argument("--db", default="knowledge_base", help="Database name")
    store_parser.add_argument("--collection", default="content", help="Collection name")
    
    # Extract entities command
    entities_parser = subparsers.add_parser("entities", help="Extract entities from content")
    entities_parser.add_argument("input", help="Input file with processed content")
    entities_parser.add_argument("--output", help="Output file for the extracted entities")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--db", default="knowledge_base", help="Database name")
    query_parser.add_argument("--collection", default="content", help="Collection name")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search the knowledge base")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--db", default="knowledge_base", help="Database name")
    search_parser.add_argument("--collection", default="content", help="Collection name")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    ensure_directories(config)
    
    # Run the command
    if args.command == "extract":
        # Extract content from a source
        result = asyncio.run(extract_source(args.source, args.type))
        
        if "error" in result:
            logger.error(result["error"])
            sys.exit(1)
        
        # Save the result
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Extraction result saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
    
    elif args.command == "process":
        # Load the input
        try:
            with open(args.input, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load input file: {str(e)}")
            sys.exit(1)
        
        # Process the content
        result = asyncio.run(process_content(
            data["content"],
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        ))
        
        if "error" in result:
            logger.error(result["error"])
            sys.exit(1)
        
        # Save the result
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Processing result saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
    
    elif args.command == "store":
        # Load the input
        try:
            with open(args.input, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load input file: {str(e)}")
            sys.exit(1)
        
        # Store the content
        result = asyncio.run(store_content(
            data["chunks"],
            db_name=args.db,
            collection_name=args.collection,
        ))
        
        if "error" in result:
            logger.error(result["error"])
            sys.exit(1)
        
        print(json.dumps(result, indent=2))
    
    elif args.command == "entities":
        # Load the input
        try:
            with open(args.input, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load input file: {str(e)}")
            sys.exit(1)
        
        # Extract entities
        result = asyncio.run(extract_entities(data["chunks"]))
        
        if "error" in result:
            logger.error(result["error"])
            sys.exit(1)
        
        # Save the result
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Entity extraction result saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
    
    elif args.command == "query":
        # Query the knowledge base
        result = asyncio.run(query_knowledge_base(
            args.query,
            db_name=args.db,
            collection_name=args.collection,
        ))
        
        if "error" in result:
            logger.error(result["error"])
            sys.exit(1)
        
        print(json.dumps(result, indent=2))
    
    elif args.command == "search":
        # Search the knowledge base
        result = asyncio.run(search_knowledge_base(
            args.query,
            db_name=args.db,
            collection_name=args.collection,
        ))
        
        if "error" in result:
            logger.error(result["error"])
            sys.exit(1)
        
        print(json.dumps(result, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 