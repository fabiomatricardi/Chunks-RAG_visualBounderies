import os
import json
from pathlib import Path
from docling.document_converter import DocumentConverter
import gradio as gr
import tiktoken
import time
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from pathlib import Path
import hashlib

# MODELS RUNNING
# .\llama-server.exe -m C:\FABIO-AI\MODELS_Small\Qwen3-0.6B-Q8_0.gguf --port 8081 -c 8192
# .\llama-server.exe -m C:\FABIO-AI\MODELS_embeddings\bge-small-en-v1.5_fp16.gguf --port 8080 --embedding

# Constants
CHUNK_DELIMITER = "<!-- CHUNK -->"
OUTPUT_DIR = Path.cwd()
ENCODING_NAME = "cl100k_base"  # Standard for GPT-3.5/4; change if needed

# Initialize tokenizer once
tokenizer = tiktoken.get_encoding(ENCODING_NAME)

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def count_prompt_tokens(system_prompt: str, context: str, query: str) -> int:
    """Count tokens in the full RAG prompt using tiktoken."""
    full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}"
    return count_tokens(full_prompt)

# Ensure output dir exists
OUTPUT_DIR.mkdir(exist_ok=True)

def list_chunk_files():
    return [f.name for f in OUTPUT_DIR.glob("*_chunks.json")]  

# ----------------------------
# Tab 1: PDF to Markdown
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


# ----------------------------
# Tab 2: Markdown Chunking
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



# Cache for embeddings: {chunk_hash: embedding_vector}
embedding_cache = {}

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """Get embedding from local llama.cpp server."""
    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]

    try:
        response = requests.post(
            "http://localhost:8080/v1/embeddings",
            json={"input": text, "model": model},
            timeout=30
        )
        response.raise_for_status()
        embedding = response.json()["data"][0]["embedding"]
        embedding_cache[cache_key] = embedding
        return embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        # Fallback: zero vector (will rank last)
        return [0.0] * 384  # adjust size if your model differs

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0  # Zero vector has no direction → similarity = 0
    
    return np.dot(a, b) / (norm_a * norm_b)

# ----------------------------
# Tab 4: BM25 Search
# ----------------------------
def list_chunk_files_for_search():
    return [f.name for f in OUTPUT_DIR.glob("*_chunks.json")]

def bm25_search(chunk_filenames, query, top_k=5):
    if not chunk_filenames or not query:
        return []

    all_chunks = []
    for filename in chunk_filenames:
        try:
            with open(OUTPUT_DIR / filename, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
        except Exception as e:
            return [{"error": f"Failed to load {filename}: {str(e)}"}]

    if not all_chunks:
        return []

    corpus = [chunk["text"].split() for chunk in all_chunks]
    bm25 = BM25Okapi(corpus)
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        chunk = all_chunks[idx].copy()  # don't mutate original
        chunk["_score_bm25"] = float(scores[idx])  # add score
        results.append(chunk)
    
    return results



# ----------------------------
# Tab 5: Semantic Search
# ----------------------------
def semantic_search(chunk_filenames, query, top_k=5):
    if not chunk_filenames or not query:
        return []

    all_chunks = []
    for filename in chunk_filenames:
        try:
            with open(OUTPUT_DIR / filename, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
        except Exception as e:
            return [{"error": f"Failed to load {filename}: {str(e)}"}]

    if not all_chunks:
        return []

    query_emb = get_embedding(query)
    results = []
    similarities = []

    for chunk in all_chunks:
        chunk_emb = get_embedding(chunk["text"])
        sim = float(cosine_similarity(query_emb, chunk_emb))
        similarities.append(sim)

    top_indices = np.argsort(similarities)[::-1][:top_k]
    for idx in top_indices:
        chunk = all_chunks[idx].copy()
        chunk["_score_semantic"] = similarities[idx]
        results.append(chunk)

    return results


# ----------------------------
# Tab 6: Hybrid Search
# ----------------------------

def hybrid_search(chunk_filenames, query, top_k=5, alpha=0.5):
    if not chunk_filenames or not query:
        return []

    all_chunks = []
    for filename in chunk_filenames:
        try:
            with open(OUTPUT_DIR / filename, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
        except Exception as e:
            return [{"error": f"Failed to load {filename}: {str(e)}"}]

    if not all_chunks:
        return []

    # BM25 scores
    corpus = [chunk["text"].split() for chunk in all_chunks]
    bm25 = BM25Okapi(corpus)
    query_tokens = query.split()
    bm25_scores = bm25.get_scores(query_tokens)
    min_b, max_b = min(bm25_scores), max(bm25_scores)
    bm25_norm = [0.5] * len(bm25_scores) if min_b == max_b else [(s - min_b) / (max_b - min_b) for s in bm25_scores]

    # Semantic scores
    query_emb = get_embedding(query)
    semantic_scores = []
    for chunk in all_chunks:
        chunk_emb = get_embedding(chunk["text"])
        sim = float(cosine_similarity(query_emb, chunk_emb))
        semantic_scores.append(sim)

    # Hybrid score
    hybrid_scores = [alpha * b + (1 - alpha) * s for b, s in zip(bm25_norm, semantic_scores)]

    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        chunk = all_chunks[idx].copy()
        chunk["_score_hybrid"] = hybrid_scores[idx]
        chunk["_bm25_norm"] = bm25_norm[idx]
        chunk["_semantic"] = semantic_scores[idx]
        results.append(chunk)

    return results

# -------------------------------
# Tab 7: FULL RAG & Hybrid Search
# -------------------------------
def rag_complete_full(chunk_filenames, query, top_k=5, alpha=0.5, system_prompt=None):
    """
    Non-streaming RAG that returns full answer + timing + chunks.
    Fully compatible with Gradio 3.x.
    """
    start_time = time.time()
    
    if not chunk_filenames or not query:
        return "⚠️ Please select chunk files and enter a question.", [], "⏳ Ready"

    # Simulate "retrieval in progress" by just computing
    retrieved_chunks = hybrid_search(chunk_filenames, query, top_k=top_k, alpha=alpha)
    if not retrieved_chunks or (isinstance(retrieved_chunks, list) and len(retrieved_chunks) > 0 and "error" in retrieved_chunks[0]):
        return "❌ Retrieval failed. Check your chunk files.", retrieved_chunks, "🚨 Retrieval error"

    context = "\n\n".join([
        f"[Source: {chunk.get('filename', 'unknown')}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    ])

    default_system = (
        "You are a helpful assistant that answers questions based ONLY on the provided context. "
        "If the context doesn't contain the answer, say 'I don't know based on the provided documents.'"
    )
    sys_msg = system_prompt or default_system

    prompt_tokens = count_prompt_tokens(sys_msg, context, query)
    
    # Call LLM (non-streaming)
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    try:
        response = requests.post(
            "http://localhost:8081/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 500,
                "stream": False  # ← Non-streaming
            },
            timeout=190
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        
        total_time = time.time() - start_time
        status = f"✅ Done in {total_time:.1f}s | 📊 Prompt: {prompt_tokens} tokens"
        return answer, retrieved_chunks, status

    except Exception as e:
        total_time = time.time() - start_time
        status = f"🚨 Failed after {total_time:.1f}s"
        return f"🚨 LLM error: {str(e)}", retrieved_chunks, status

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

            selected_filename = gr.State("")  # to track current file

            def on_file_select(filename):
                content, _ = load_markdown_file(filename)
                return content, content, filename

            md_file_dropdown.change(
                fn=on_file_select,
                inputs=md_file_dropdown,
                outputs=[editable_md, original_md_preview, selected_filename]
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
        # ---------------- Tab 3: Explore Chunk Files ----------------
        with gr.Tab("📊 Explore Chunk Files"):
            with gr.Row():       
                chunk_file_dropdown = gr.Dropdown(
                    choices=list_chunk_files(),
                    label="Select Chunk JSON File",
                    interactive=True,
                    allow_custom_value=True
                )
                refresh_chunk_btn = gr.Button("🔄 Refresh List")

            chunk_json_viewer = gr.JSON(label="Chunks (click to expand)")

            def load_chunks_as_json(filename):
                if isinstance(filename, list):
                    filename = filename[0] if filename else ""
                if not filename or not isinstance(filename, str):
                    return {}
                try:
                    with open(OUTPUT_DIR / filename, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    return {"error": str(e)}

            # Remove the demo.load() line!

            refresh_chunk_btn.click(
                fn=list_chunk_files,
                outputs=chunk_file_dropdown
            )

            chunk_file_dropdown.change(
                fn=load_chunks_as_json,
                inputs=chunk_file_dropdown,
                outputs=chunk_json_viewer
            )

        # ---------------- Tab 4: BM25 Search ----------------
        with gr.Tab("🔍 BM25 Search"):
            with gr.Row():
                bm25_file_dropdown = gr.Dropdown(
                    choices=list_chunk_files_for_search(),
                    label="Select Chunk JSON Files",
                    multiselect=True,
                    interactive=True
                )
                bm25_refresh_btn = gr.Button("🔄 Refresh List")
            bm25_query = gr.Textbox(label="Query", placeholder="Enter your search query...")
            bm25_top_k = gr.Slider(1, 20, value=5, step=1, label="Top K Results")
            bm25_search_btn = gr.Button("🔍 Search with BM25")
            bm25_output = gr.JSON(label="BM25 Results")
            bm25_search_btn.click(
                fn=bm25_search,
                inputs=[bm25_file_dropdown, bm25_query, bm25_top_k],
                outputs=bm25_output
            )

            bm25_refresh_btn.click(
                fn=list_chunk_files_for_search,
                outputs=bm25_file_dropdown
            )


        # ---------------- Tab 5: Semantic Search ----------------
        with gr.Tab("🧠 Semantic Search"):
            with gr.Row():
                sem_file_dropdown = gr.Dropdown(
                    choices=list_chunk_files_for_search(),
                    label="Select Chunk JSON Files",
                    multiselect=True,
                    interactive=True
                )
                sem_refresh_btn = gr.Button("🔄 Refresh List")
            sem_query = gr.Textbox(label="Query", placeholder="Enter your search query...")
            sem_top_k = gr.Slider(1, 20, value=5, step=1, label="Top K Results")
            sem_search_btn = gr.Button("🧠 Search with Embeddings")
            sem_output = gr.JSON(label="Semantic Search Results")
            sem_search_btn.click(
                fn=semantic_search,
                inputs=[sem_file_dropdown, sem_query, sem_top_k],
                outputs=sem_output
            )

            sem_refresh_btn.click(
                fn=list_chunk_files_for_search,
                outputs=sem_file_dropdown
            )


        # ---------------- Tab 6: Hybrid Search ----------------
        with gr.Tab("⚡ Hybrid Search"):
            with gr.Row():
                hybrid_file_dropdown = gr.Dropdown(
                    choices=list_chunk_files_for_search(),
                    label="Select Chunk JSON Files",
                    multiselect=True,
                    interactive=True
                )
                hybrid_refresh_btn = gr.Button("🔄 Refresh List")
            hybrid_query = gr.Textbox(label="Query", placeholder="Enter your search query...")
            with gr.Row():
                hybrid_top_k = gr.Slider(1, 20, value=5, step=1, label="Top K Results")
                hybrid_alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="BM25 Weight (α)")
            hybrid_search_btn = gr.Button("⚡ Search with Hybrid")
            hybrid_output = gr.JSON(label="Hybrid Search Results")
            hybrid_search_btn.click(
                fn=hybrid_search,
                inputs=[hybrid_file_dropdown, hybrid_query, hybrid_top_k, hybrid_alpha],
                outputs=hybrid_output
            )

            hybrid_refresh_btn.click(
                fn=list_chunk_files_for_search,
                outputs=hybrid_file_dropdown
            )

        # ---------------- Tab 7: Full RAG with LLM (Streaming + Token Count) ----------------
        with gr.Tab("🤖 Full RAG (Hybrid + LLM)"):
            with gr.Row():
                rag_file_dropdown = gr.Dropdown(
                    choices=list_chunk_files_for_search(),
                    label="Select Chunk JSON Files",
                    multiselect=True,
                    interactive=True
                )
                rag_refresh_btn = gr.Button("🔄 Refresh List")
            
            rag_query = gr.Textbox(label="Question", placeholder="Ask a question about your documents...")
            
            with gr.Row():
                rag_top_k = gr.Slider(1, 20, value=5, step=1, label="Top K Chunks")
                rag_alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="BM25 Weight (α)")
            
            rag_system_prompt = gr.Textbox(
                label="System Prompt (optional)",
                placeholder="Customize LLM behavior...",
                lines=2
            )
            
            rag_btn = gr.Button("🤖 Get Answer")
            rag_token_count = gr.Textbox(label="Token Usage", interactive=False)
            rag_answer = gr.Textbox(label="Answer", lines=8, interactive=False)
            rag_context = gr.JSON(label="Retrieved Chunks (for verification)")
            
            # Refresh file list
            rag_refresh_btn.click(
                fn=list_chunk_files_for_search,
                outputs=rag_file_dropdown
            )
            
            # Connect streaming function
            rag_btn.click(
                fn=rag_complete_full,  # ← Non-streaming
                inputs=[rag_file_dropdown, rag_query, rag_top_k, rag_alpha, rag_system_prompt],
                outputs=[rag_answer, rag_context, rag_token_count]
            )

# Launch
if __name__ == "__main__":
    demo.launch()