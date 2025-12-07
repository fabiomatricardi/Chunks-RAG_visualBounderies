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
from typing import Optional
import tiktoken


# Constants
OUTPUT_DIR = Path.cwd()
ENCODING_NAME = "cl100k_base"
CHUNK_DELIMITER = "<!-- CHUNK -->"
API_BASE = "http://localhost:8080/v1"
# Global server variable
LLMserver = None

# Initialize tokenizer once
tokenizer = tiktoken.get_encoding(ENCODING_NAME)

# ----------------------------
# TAB 1 Functions  
# ----------------------------

def convert_pdf_to_md(pdf_file):
    if pdf_file is None:
        return "❌ No file uploaded."
  
    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_file.name)
        md_content = result.document.export_to_markdown()

        # Derive output filename
        md_filename = Path(pdf_file.name).with_suffix('.md').name
        md_path = OUTPUT_DIR / md_filename

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return f"✅ Successfully converted and saved as `{md_filename}`"
    except Exception as e:
        return f"❌ Error: {str(e)}"
    
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# ----------------------------
# TAB 2 Functions  
# ----------------------------

def list_markdown_files():
    return [f.name for f in OUTPUT_DIR.glob("*.md")]

def load_markdown_file(filename):
    if not filename:
        return "", ""
    try:
        with open(OUTPUT_DIR / filename, "r", encoding="utf-8") as f:
            content = f.read()
        return content, content
    except Exception as e:
        return f"Error loading file: {e}", ""

def update_token_count(md_text):
    if not md_text:
        return "0"
    return str(count_tokens(md_text))

def process_chunks(md_text, filename):
    if not md_text or not filename:
        return "⚠️ No content or filename.", None

    raw_chunks = md_text.split(CHUNK_DELIMITER)
    chunks = []
    total_tokens = 0
    preview_lines = []

    for i, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        token_count = count_tokens(chunk_text)
        total_tokens += token_count

        chunk_meta = {
            "filename": filename,
            "chunk_index": i,
            "token_count": token_count,
            "text": chunk_text
        }
        chunks.append(chunk_meta)

        # Preview line: show first 100 chars + token count
        preview = f"[Chunk {i}] ({token_count} tokens)\n{chunk_text[:100]}{'...' if len(chunk_text) > 100 else ''}"
        preview_lines.append(preview)

    # Build preview display
    display_text = (
        f"✅ Total chunks: {len(chunks)} | Total tokens: {total_tokens}\n\n"
        + ("\n" + "="*60 + "\n").join(preview_lines)
    )

    # Save as JSON
    json_path = OUTPUT_DIR / f"{Path(filename).stem}_chunks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    return display_text, str(json_path)

# ------------------------------------
# TAB 2  - Server Management Functions
# ------------------------------------
def is_server_ready(timeout: int = 300, interval: int = 3) -> bool:
    """Check if the server is ready to accept requests"""
    print("Checking server readiness...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # First check specific endpoint
            response = requests.get(f"{API_BASE}/models", timeout=3)
            if response.status_code == 200:
                models = response.json()
                if "data" in models and len(models["data"]) > 0:
                    print(f"✅ Server is ready! Models loaded: {len(models['data'])}")
                    return True
                print(f"⚠️ Server responded but no models loaded: {models}")
            else:
                print(f"⚠️ Server responded with status: {response.status_code}")
        
        except requests.exceptions.ConnectionError as e:
            print(f"⌛ Still connecting to server... ({str(e)[:50]}...)")
        except Exception as e:
            print(f"⏳ Waiting for server... ({str(e)[:50]}...)")
        
        # Show elapsed time
        elapsed = int(time.time() - start_time)
        print(f"   Time elapsed: {elapsed}s / {timeout}s")
        time.sleep(interval)
    
    print(f"❌ Server not ready after {timeout} seconds")
    return False


def start_llm_server(visible=True):
    global LLMserver
    
    # Check if process is already running
    if LLMserver is not None:
        if LLMserver.poll() is None:
            return f"⚠️ Server is already running (PID: {LLMserver.pid})"
        else:
            LLMserver = None  # Clean up terminated reference
    
    model_path = 'gemma-3-270m-it-Q8_0.gguf'
    if not os.path.exists(model_path):
        return f"❌ Model file not found: {model_path}"
    
    # Start server with better logging
    try:
        LLMserver = subprocess.Popen(
            [
                'llama-server.exe',
                '-m', model_path,
                '-c', '12000',
                '-ngl', '0',
                '--port', '8080',
                '--temp', '0.3',
                '--repeat-penalty', '1.45',
                '--verbose'
            ], 
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Wait for model loading
        time.sleep(5)
        status = f"Server started (PID: {LLMserver.pid}). Loading model..."
        print(status)
        return status
    except Exception as e:
        return f"❌ Error starting server: {str(e)}"

def stop_llm_server():
    global LLMserver
    
    if LLMserver is None:
        return "No server is running!"
    
    if LLMserver.poll() is None:
        try:
            LLMserver.terminate()
            LLMserver.wait(timeout=5)
            return "Server stopped successfully!"
        except subprocess.TimeoutExpired:
            LLMserver.kill()
            return "Server killed forcefully!"
    else:
        return "Server is not running!"


# ----------------------------
# TAB 2  - Analysis Functions
# ----------------------------
def genOS_chat(client, user_prompt, history, stops):
    """Send prompt to local model and get response - with better error handling"""
    try:
        history.append({"role": "user", "content": user_prompt})

        completion = client.chat.completions.create(
            model="local-model",
            messages=history,
            temperature=0.3,
            frequency_penalty=1.45,
            max_tokens=800,
            stop=stops
        )
        response = completion.choices[0].message.content
        history.append({"role": "assistant", "content": response})
        return history
    except Exception as e:
        print(f"Chat API error: {str(e)}")
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return history

def analyze_document(text: str) -> tuple[str, str]:
    """Get summary and topics from the document via LLM - using full text"""
    if not text.strip():
        return "No content to analyze", ""
    
    try:
        client = openai.OpenAI(
            base_url=API_BASE,
            api_key="not-required"
        )
        
        # First get summary (using full text)
        summary_prompt = f"""Write a short, concise summary of the following article in 3-5 sentences:

[start of article]
{text}
[end of article]

Summary: """
        
        summary_history = []
        summary_history = genOS_chat(client, summary_prompt, summary_history, ['<eos>'])
        summary = summary_history[-1]["content"] if len(summary_history) > 1 else "No summary generated"
        print(summary)  #only for debug
        
        # Then get topics (using full text)
        topics_prompt = f"""write the 5 most relevant topics from the provided article."""
        
        topics_history = [] # Cahining the messags to use KV cache and speed up generation
        topics_history = genOS_chat(client, topics_prompt, summary_history, ['<eos>'])
        topics = topics_history[-1]["content"] if len(topics_history) > 1 else "No topics extracted"
        print(topics)  #only for debug
        
        return summary, topics
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return f"Analysis error: {str(e)}", f"Topics error: {str(e)}"

# -------------------------------
# TAB 2  - Analysis with Progress
# -------------------------------
def process_for_analysis(md_text, progress):
    """Process the document for analysis - with better diagnostics"""
    try:
        if not md_text or not md_text.strip():
            return "⚠️ No content to analyze.", "", ""
        
        # Update status
        progress(0.1, desc="Starting document analysis...")
        yield "🔧 Starting LLM server...", "", ""
        
        # Start server
        server_msg = start_llm_server(visible=True)
        yield f"🔄 {server_msg}", "", ""
        
        # Wait for server with progress
        progress(0.2, desc="Waiting for server initialization...")
        start_time = time.time()
        
        while not is_server_ready():
            elapsed = int(time.time() - start_time)
            status = f"⏳ Server is initializing... ({elapsed}s)"
            print(f"Status: {status}")
            yield status, "", ""
            time.sleep(2)
            
            if elapsed > 90:  # 90 second timeout
                yield "❌ Server took too long to initialize. Check terminal window for errors.", "", ""
                return
        
        # Verify API connection
        progress(0.3, desc="Verifying API connection...")
        try:
            client = openai.OpenAI(base_url=API_BASE, api_key="not-required")
            client.models.list()
            yield "✅ Server ready! Starting analysis...", "", ""
        except Exception as e:
            yield f"❌ API connection failed: {str(e)}", "", ""
            return
        
        # Get summary
        progress(0.5, desc="Generating summary...")
        yield "📝 Generating document summary...", "", ""
        
        summary, _ = analyze_document(md_text)
        yield "✅ Summary generated!", summary, ""
        
        # Get topics
        progress(0.8, desc="Extracting main topics...")
        yield "📊 Extracting main topics...", "", ""
        
        _, topics = analyze_document(md_text)
        
        # Final success
        progress(1.0, desc="Analysis complete!")
        yield "✅ Analysis completed successfully!", summary, topics
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_msg = f"❌ Error: {str(e)}\n\nDetails:\n{error_details}"
        print(error_msg)
        yield error_msg, "", ""

