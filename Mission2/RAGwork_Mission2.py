import os
import json
import subprocess
import threading
import time
import requests
import openai
from pathlib import Path
from docling.document_converter import DocumentConverter
import gradio as gr
import tiktoken
from typing import Optional
from RAGworkLIB import is_server_ready, start_llm_server, stop_llm_server #TAB2 server functions
from RAGworkLIB import genOS_chat, analyze_document, process_for_analysis #TAB2 LLM generation functions
from RAGworkLIB import list_markdown_files, load_markdown_file, update_token_count, process_chunks
from RAGworkLIB import convert_pdf_to_md, count_tokens  #TAB1 function

# Constants
CHUNK_DELIMITER = "<!-- CHUNK -->"
OUTPUT_DIR = Path.cwd()
ENCODING_NAME = "cl100k_base"
API_BASE = "http://localhost:8080/v1"

# Global server variable
LLMserver = None

# ----------------------------
# Gradio Interface
# ----------------------------
with gr.Blocks(title="PDF to Markdown & Chunk Editor") as demo:
    gr.Markdown("# 📄 PDF to Markdown Converter & Chunk Editor")

    with gr.Tabs():
        # ---------------- Tab 1 ----------------
        with gr.Tab("📄 Convert PDF to Markdown"):
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            convert_btn = gr.Button("Convert to Markdown")
            convert_output = gr.Textbox(label="Result", interactive=False)

            convert_btn.click(
                fn=convert_pdf_to_md,
                inputs=pdf_input,
                outputs=convert_output
            )

        # ---------------- Tab 2 ----------------
        with gr.Tab("✂️ Chunk Editor"):
            with gr.Row():
                # Left column for stats and analysis
                with gr.Column(scale=1):
                    total_tokens_display = gr.Textbox(label="Total Document Tokens", interactive=False)
                    summarize_btn = gr.Button("🔍 Start Document Analysis")
                    server_status = gr.Textbox(label="Server Status", interactive=False, value="Server not started")
                    analysis_status = gr.Textbox(label="Analysis Status", interactive=False, value="Ready to analyze")
                    summary_output = gr.TextArea(label="Document Summary", lines=10)
                    topics_output = gr.TextArea(label="Main Topics (5)", lines=7, value="", placeholder="Topics will appear here after analysis")
                    stop_server_btn = gr.Button("🛑 Stop Server", variant="stop")
                
                # Right column for file selection and editing
                with gr.Column(scale=2):
                    with gr.Row():
                        md_file_dropdown = gr.Dropdown(
                            choices=list_markdown_files(),
                            label="Select Markdown File",
                            interactive=True
                        )
                        refresh_btn = gr.Button("🔄 Refresh List")

                    with gr.Row():
                        original_md_preview = gr.Markdown(label="Original Preview")
                        editable_md = gr.Textbox(
                            label="Editable Markdown (Insert chunk breaks with <!-- CHUNK -->)",
                            lines=20,
                            max_lines=30
                        )

                    selected_filename = gr.State()  # to track current file

                    def on_file_select(filename):
                        content, _ = load_markdown_file(filename)
                        token_count = update_token_count(content)
                        return content, content, filename, token_count

                    md_file_dropdown.change(
                        fn=on_file_select,
                        inputs=md_file_dropdown,
                        outputs=[editable_md, original_md_preview, selected_filename, total_tokens_display]
                    )

                    refresh_btn.click(
                        fn=lambda: gr.update(choices=list_markdown_files()),
                        inputs=None,
                        outputs=md_file_dropdown
                    )

                    process_btn = gr.Button("✂️ Process Chunks")
                    chunks_output = gr.Textbox(label="Processed Chunks (Preview)", lines=10)
                    download_json = gr.File(label="Download Chunks as JSON", visible=True)

                    process_btn.click(
                        fn=process_chunks,
                        inputs=[editable_md, selected_filename],
                        outputs=[chunks_output, download_json]
                    )
            
            # Connect analysis button
            def analysis_wrapper(doc_text):
                # Create a progress object
                progress = gr.Progress()
                # Process with progress and yield the results
                for status, summary, topics in process_for_analysis(doc_text, progress):
                    yield status, summary, topics
            
            summarize_btn.click(
                fn=analysis_wrapper,
                inputs=editable_md,
                outputs=[analysis_status, summary_output, topics_output],
                api_name="analyze_document"
            )

            # Connect stop server button
            stop_server_btn.click(
                fn=stop_llm_server,
                outputs=server_status
            )

            # Start server status update
            def server_status_check():
                global LLMserver
                if LLMserver is None:
                    return "Server not started"
                
                if LLMserver.poll() is None:
                    return f"Server running (PID: {LLMserver.pid})"
                else:
                    return "Server stopped"

            # Update server status periodically
            demo.load(
                fn=lambda: gr.Textbox(value=server_status_check, interactive=False),
                outputs=server_status
            )

# Launch
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    demo.launch()
