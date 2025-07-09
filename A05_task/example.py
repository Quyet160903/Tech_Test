#!/usr/bin/env python
"""
Example script demonstrating how to use the Knowledge Base System
This example shows the complete workflow: extract → process → store → query
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_base.extractors.extraction_manager import ExtractionManager
from knowledge_base.processors.text_processor import TextProcessor
from knowledge_base.storage.vector_storage import VectorStorage
from knowledge_base.storage.mongodb_storage import MongoDBStorage
from knowledge_base.storage.memory_storage import MemoryStorage
from knowledge_base.agent.openai_agent import OpenAIAgent
from knowledge_base.search.semantic_search import SemanticSearch

from knowledge_base.processors import ProcessorConfig
from knowledge_base.storage import StorageConfig
from knowledge_base.agent import AgentConfig
from knowledge_base.search import SearchConfig

from knowledge_base.utils.logging import logger


async def main():
    """Main example function demonstrating the complete workflow"""
    
    # Load configuration
    load_dotenv()
    
    logger.info("🚀 Starting Knowledge Base System Example")
    logger.info("=" * 60)
    
    # Initialize components
    extraction_manager = ExtractionManager()
    text_processor = TextProcessor()
    
    # Storage backends
    vector_storage = VectorStorage()
    mongo_storage = MongoDBStorage()
    memory_storage = MemoryStorage()  # Fallback
    
    # AI components
    openai_agent = OpenAIAgent(
        vector_storage=vector_storage,
        mongo_storage=mongo_storage,
        memory_storage=memory_storage
    )
    semantic_search = SemanticSearch(
        vector_storage=vector_storage,
        metadata_storage=mongo_storage
    )
    
    try:
        # Step 1: Initialize storage systems
        logger.info("📦 Step 1: Initializing storage systems")
        
        # Always initialize memory storage as fallback
        await memory_storage.initialize(StorageConfig(
            database_name="knowledge_base",
            collection_name="content"
        ))
        logger.info("✅ Memory storage initialized")
        
        # Try to initialize MongoDB
        mongo_success = await mongo_storage.initialize(StorageConfig(
            database_name="knowledge_base",
            collection_name="content"
        ))
        if mongo_success:
            logger.info("✅ MongoDB storage initialized")
        else:
            logger.warning("⚠️  MongoDB not available, using memory storage")
        
        # Try to initialize vector storage
        vector_success = await vector_storage.initialize(StorageConfig(
            database_name="knowledge_base",
            collection_name="content_vectors"
        ))
        if vector_success:
            logger.info("✅ Vector storage (ChromaDB) initialized")
        else:
            logger.warning("⚠️  ChromaDB not available, using memory storage")
        
        # Step 2: Add and extract from a source
        logger.info("\n🌐 Step 2: Adding and extracting from a source")
        
        # Add a simple Wikipedia source
        source_id = "example_wikipedia"
        source_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        
        logger.info(f"📥 Adding source: {source_id}")
        logger.info(f"🔗 URL: {source_url}")
        
        success = await extraction_manager.add_source(
            source_id=source_id,
            location=source_url,
            source_type="web",
            metadata={"example": True, "topic": "programming"}
        )
        
        if not success:
            logger.error("❌ Failed to add source")
            return
        
        logger.info("✅ Source added successfully")
        
        # Extract content
        logger.info("🔄 Extracting content...")
        extraction_result = await extraction_manager.extract_source(source_id)
        
        if extraction_result.error:
            logger.error(f"❌ Extraction error: {extraction_result.error}")
            return
        
        logger.info(f"✅ Extracted {len(extraction_result.content)} content items")
        
        # Step 3: Process the content
        logger.info("\n📝 Step 3: Processing content")
        
        processing_result = await text_processor.process(
            extraction_result.content,
            ProcessorConfig(chunk_size=1000, chunk_overlap=200)
        )
        
        if processing_result.error:
            logger.error(f"❌ Processing error: {processing_result.error}")
            return
        
        logger.info(f"✅ Generated {len(processing_result.content_chunks)} content chunks")
        
        # Step 4: Store the content
        logger.info("\n💾 Step 4: Storing content")
        
        # Try MongoDB first, fallback to memory
        mongo_result = await mongo_storage.store(processing_result.content_chunks)
        if not mongo_result.success:
            logger.warning("⚠️  MongoDB storage failed, using memory storage")
            mongo_result = await memory_storage.store(processing_result.content_chunks)
        
        # Try vector storage, fallback to memory
        vector_result = await vector_storage.store(processing_result.content_chunks)
        if not vector_result.success:
            logger.warning("⚠️  Vector storage failed, using memory storage")
            vector_result = await memory_storage.store(processing_result.content_chunks)
        
        logger.info(f"✅ Stored {mongo_result.count} documents in metadata storage")
        logger.info(f"✅ Stored {vector_result.count} documents in vector storage")
        
        # Step 5: Initialize AI agent
        logger.info("\n🤖 Step 5: Initializing AI agent")
        
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("⚠️  OPENAI_API_KEY not found in environment")
            logger.info("💡 Please set your OpenAI API key to test the AI features")
            logger.info("   export OPENAI_API_KEY='your-key-here'")
            logger.info("\n✅ Example completed (without AI features)")
            return
        
        agent_success = await openai_agent.initialize(AgentConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500
        ))
        
        if not agent_success:
            logger.error("❌ Failed to initialize AI agent")
            return
        
        logger.info("✅ AI agent initialized")
        
        # Step 6: Query the knowledge base
        logger.info("\n🔍 Step 6: Querying the knowledge base")
        
        queries = [
            "What is Python programming language?",
            "What are the main features of Python?",
            "How is Python used in data science?"
        ]
        
        for query in queries:
            logger.info(f"\n❓ Query: {query}")
            
            response = await openai_agent.query(query)
            
            if response.error:
                logger.error(f"❌ Query error: {response.error}")
                continue
            
            logger.info(f"🤖 Answer: {response.answer}...")
            logger.info(f"📚 Sources found: {len(response.sources)}")
            logger.info(f"🏷️  Related entities: {len(response.related_entities)}")
            logger.info(f"📊 Confidence: {response.confidence:.1%}")
        
        # Step 7: Semantic search
        logger.info("\n🔎 Step 7: Semantic search")
        
        search_success = await semantic_search.initialize(SearchConfig())
        if not search_success:
            logger.warning("⚠️  Failed to initialize semantic search")
        else:
            search_queries = ["object-oriented programming", "data types", "libraries"]
            
            for search_query in search_queries:
                logger.info(f"\n🔍 Searching for: {search_query}")
                
                search_result = await semantic_search.search(
                    query=search_query,
                    limit=3
                )
                
                if search_result.error:
                    logger.error(f"❌ Search error: {search_result.error}")
                    continue
                
                logger.info(f"📊 Found {search_result.total_count} results")
                
                # Show top result
                if search_result.items:
                    top_result = search_result.items[0]
                    relevance = top_result.get('relevance_score', 0.0)
                    text_preview = top_result.get('text', '')[:150]
                    logger.info(f"🥇 Top result (relevance: {relevance:.2f}): {text_preview}...")
        
        # Step 8: Show statistics
        logger.info("\n📊 Step 8: System statistics")
        
        stats = extraction_manager.get_extraction_stats()
        logger.info(f"📋 Total sources: {stats.get('total_sources', 0)}")
        logger.info(f"✅ Successful extractions: {stats.get('successful_extractions', 0)}")
        logger.info(f"❌ Failed extractions: {stats.get('failed_extractions', 0)}")
        
    except Exception as e:
        logger.error(f"❌ Error during example execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 