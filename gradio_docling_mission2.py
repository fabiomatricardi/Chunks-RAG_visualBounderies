import os
import json
from pathlib import Path
from docling.document_converter import DocumentConverter
import gradio as gr
import tiktoken

# Constants
CHUNK_DELIMITER = "<!-- CHUNK -->" ##
OUTPUT_DIR = Path.cwd()
ENCODING_NAME = "cl100k_base"  ## Standard for GPT-3.5/4; change if needed

## Initialize tokenizer once
tokenizer = tiktoken.get_encoding(ENCODING_NAME)

## new function to count tokens
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Ensure output dir exists
OUTPUT_DIR.mkdir(exist_ok=True)

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

# Launch
if __name__ == "__main__":
    demo.launch()