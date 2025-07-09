"""
Gradio GUI for Knowledge Base System
Main workflow: Add Source → Auto Extract → Query
"""

import gradio as gr
import requests
import json
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Base URL
API_BASE_URL = "http://localhost:8000/api"

class KnowledgeBaseGUI:
    def __init__(self):
        self.sources = []
        
    def check_health(self):
        """Check if backend is running"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                return "✅ Backend is running"
            else:
                return "❌ Backend not responding"
        except Exception as e:
            return f"❌ Backend not available: {str(e)}"
    
    def add_source_and_extract(self, source_id: str, location: str, source_type: str):
        """Add source and automatically extract content"""
        if not source_id or not location:
            return "❌ Error: Source ID and Location are required", ""
        
        try:
            # Add source with automatic extraction
            payload = {
                "source_id": source_id,
                "location": location,
                "source_type": source_type,
                "metadata": {}
            }
            
            response = requests.post(
                f"{API_BASE_URL}/sources", 
                json=payload,
                timeout=120  # 2 minutes for extraction
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Format success message
                if result.get("storage_type") == "memory":
                    message = f"""✅ Source processed successfully!
📊 Extracted: {result.get('extracted_items', 0)} items
🔄 Processed: {result.get('processed_chunks', 0)} chunks
💾 Storage: Memory (databases not available)
🆔 Source ID: {source_id}"""
                else:
                    message = f"""✅ Source processed successfully!
📊 Extracted: {result.get('extracted_items', 0)} items
🔄 Processed: {result.get('processed_chunks', 0)} chunks
💾 MongoDB: {result.get('mongo_stored', 0)} chunks
🔍 ChromaDB: {result.get('vector_stored', 0)} chunks
🆔 Source ID: {source_id}"""
                
                # Get updated sources list
                sources_list = self.get_sources_list()
                
                return message, sources_list
            else:
                error_detail = response.json().get("detail", "Unknown error")
                return f"❌ Error: {error_detail}", ""
                
        except requests.exceptions.Timeout:
            return "⏱️ Timeout: Extraction is taking longer than expected. Check backend logs.", ""
        except Exception as e:
            return f"❌ Error: {str(e)}", ""
    
    def get_sources_list(self):
        """Get formatted list of sources"""
        try:
            response = requests.get(f"{API_BASE_URL}/sources", timeout=10)
            if response.status_code == 200:
                data = response.json()
                sources = data.get("sources", [])
                
                if not sources:
                    return "No sources added yet."
                
                sources_text = "📋 **Sources List:**\n\n"
                for source in sources:
                    sources_text += f"""🆔 **{source['source_id']}**
📍 Location: {source['location']}
📂 Type: {source['source_type']}
📊 Status: {source['status']}
🔢 Extractions: {source['extraction_count']}
📅 Last: {source.get('last_extracted', 'Never')}
---
"""
                return sources_text
            else:
                return "❌ Failed to load sources"
        except Exception as e:
            return f"❌ Error loading sources: {str(e)}"
    
    def query_knowledge_base(self, query: str):
        """Query the knowledge base"""
        if not query.strip():
            return "❌ Please enter a query", "", ""
        
        try:
            payload = {"query": query, "limit": 10}
            response = requests.post(
                f"{API_BASE_URL}/query",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Format answer
                answer = f"🤖 **Answer:**\n{result.get('answer', 'No answer available')}"
                
                # Format sources
                sources = result.get('sources', [])
                sources_text = "\n\n📚 **Sources:**\n"
                if sources:
                    for i, source in enumerate(sources[:3], 1):  # Top 3 sources
                        sources_text += f"{i}. {source.get('title', 'Unknown')} (Score: {source.get('relevance_score', 0):.2f})\n"
                        if source.get('url'):
                            sources_text += f"   🔗 {source['url']}\n"
                else:
                    sources_text += "No sources found\n"
                
                # Format entities
                entities = result.get('related_entities', [])
                entities_text = "\n\n🏷️ **Related Concepts:**\n"
                if entities:
                    for entity in entities[:5]:  # Top 5 entities
                        entities_text += f"• {entity.get('name', 'Unknown')} ({entity.get('entity_type', 'Unknown')})\n"
                else:
                    entities_text += "No related concepts found\n"
                
                confidence = result.get('confidence', 0)
                confidence_text = f"\n\n📊 **Confidence:** {confidence:.1%}"
                
                return answer, sources_text, entities_text + confidence_text
                
            else:
                error_detail = response.json().get("detail", "Unknown error")
                return f"❌ Query Error: {error_detail}", "", ""
                
        except Exception as e:
            return f"❌ Error: {str(e)}", "", ""
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(title="Knowledge Base System", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🧠 Knowledge Base System")
            gr.Markdown("Add sources, extract content automatically, and query your knowledge base")
            
            # Health check
            with gr.Row():
                health_btn = gr.Button("🔍 Check Backend Status", variant="secondary")
                health_output = gr.Textbox(label="Status", interactive=False)
            
            health_btn.click(self.check_health, outputs=[health_output])
            
            # Main tabs
            with gr.Tabs():
                # Tab 1: Add Sources
                with gr.TabItem("📥 Add Source"):
                    gr.Markdown("### Add a new source and automatically extract content")
                    
                    with gr.Row():
                        with gr.Column():
                            source_id_input = gr.Textbox(
                                label="Source ID", 
                                placeholder="e.g., wiki_python, pdf_manual",
                                info="Unique identifier for this source"
                            )
                            location_input = gr.Textbox(
                                label="Location (URL or Path)", 
                                placeholder="https://example.com or /path/to/file.pdf",
                                info="URL or file path of the data source"
                            )
                            source_type_input = gr.Dropdown(
                                label="Source Type",
                                choices=[
                                    "web", "pdf", "json", "csv", "xml", "rss", "txt", "docx", 
                                    "youtube", "github", "arxiv", "api"
                                ],
                                value="web",
                                info="Type of content source"
                            )
                            add_btn = gr.Button("🚀 Add Source & Extract", variant="primary")
                        
                        with gr.Column():
                            add_output = gr.Textbox(
                                label="Result", 
                                lines=8,
                                interactive=False
                            )
                    
                    sources_display = gr.Textbox(
                        label="📋 Current Sources",
                        lines=6,
                        interactive=False
                    )
                    
                    # Auto-load sources on startup
                    demo.load(self.get_sources_list, outputs=[sources_display])
                    
                    add_btn.click(
                        self.add_source_and_extract,
                        inputs=[source_id_input, location_input, source_type_input],
                        outputs=[add_output, sources_display]
                    )
                
                # Tab 2: Query Knowledge Base
                with gr.TabItem("🔍 Query"):
                    gr.Markdown("### Ask questions about your knowledge base")
                    
                    with gr.Row():
                        with gr.Column():
                            query_input = gr.Textbox(
                                label="Your Question",
                                placeholder="What is Python? How does machine learning work?",
                                lines=2
                            )
                            query_btn = gr.Button("🔎 Search Knowledge Base", variant="primary")
                        
                        with gr.Column():
                            answer_output = gr.Textbox(
                                label="Answer",
                                lines=6,
                                interactive=False
                            )
                    
                    with gr.Row():
                        sources_output = gr.Textbox(
                            label="Sources",
                            lines=4,
                            interactive=False
                        )
                        entities_output = gr.Textbox(
                            label="Related Info",
                            lines=4,
                            interactive=False
                        )
                    
                    query_btn.click(
                        self.query_knowledge_base,
                        inputs=[query_input],
                        outputs=[answer_output, sources_output, entities_output]
                    )
                    
                    # Example queries
                    gr.Markdown("### 💡 Example Queries:")
                    example_queries = [
                        "What is the main topic of the content?",
                        "Summarize the key points",
                        "What are the most important concepts?",
                        "How does this relate to [topic]?"
                    ]
                    
                    for example in example_queries:
                        gr.Button(example, variant="secondary").click(
                            lambda q=example: q, outputs=[query_input]
                        )
            
            # Footer
            gr.Markdown("---")
            gr.Markdown("🚀 **Main Workflow:** Add Source → Auto Extract → Query | ✨ Powered by Gradio")
        
        return demo

def main():
    """Main function to run the Gradio app"""
    gui = KnowledgeBaseGUI()
    demo = gui.create_interface()
    
    print("🚀 Starting Knowledge Base GUI...")
    print("📡 Backend should be running on http://localhost:8000")
    print("🖥️  GUI will be available on http://localhost:7860")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main() 