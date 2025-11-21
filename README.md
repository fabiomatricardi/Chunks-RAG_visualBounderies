# Chunks-RAG_visualBounderies
gradio app to load a pdf to markdown and manually set chunks limits

The idea behind the project is that data chunks are the right place to enhance your RAG strategy

And there is no one better than you to set those limits

Thinking in RAG, also means that you can start writing with LLM accessibility in mind, following the mindset of  [.txt](https://llmstxt.org/) project

> A proposal to standardise on using an /llms.txt file to provide information to help LLMs use a website at inference time.

### Requirements
```bash
pip install docling requests easygui gradio
pip install tiktoken
```

To achieve your goal, we’ll build a **Gradio app** with two tabs:

---

### ✅ Features Summary:

**Tab 1: PDF → Markdown**
- Upload a PDF file.
- Convert it to Markdown using `docling`.
- Save the Markdown file in the current working directory with the same name (`.md` extension).
- Show a success message.

**Tab 2: Markdown Chunking Interface**
- List all `.md` files in the current directory.
- Load and preview the selected Markdown file.
- Allow the user to **manually insert chunk delimiters** (e.g., `<!-- CHUNK -->`) directly in a text editor.
- Provide a **"Process Chunks"** button that:
  - Splits the text on the delimiter.
  - Returns a list of chunks with metadata (`filename`, `chunk_index`, etc.).
  - Shows the chunks in a structured format (e.g., JSON or list).
- Allow **saving** the chunk list as a `.json` file.

---

### 📦 Requirements

Install these if not already:

```bash
pip install gradio docling easygui
```

> Note: `easygui` is not needed in Gradio (we’ll use Gradio components instead).

---

### 🧠 Design Decisions

- **Chunk delimiter**: We’ll use a special comment like `<!-- CHUNK -->`, which is Markdown-safe and unlikely to appear in real content.
- **Chunk metadata**: Each chunk will be a dict like:
  ```python
  {
    "filename": "paper.md",
    "chunk_index": 0,
    "text": "..."
  }
  ```
- **Saving chunks**: We'll let the user download the list as JSON.

---

### 💻 Full Gradio App Code
See below after enhanchement


---

### 📁 Output Example (`paper_chunks.json`):

```json
[
  {
    "filename": "paper.md",
    "chunk_index": 0,
    "text": "Introduction...\n..."
  },
  {
    "filename": "paper.md",
    "chunk_index": 1,
    "text": "Related Work...\n..."
  }
]
```

---

### Adding tiktoken KPI

Adding **token count per chunk** using `tiktoken` is a smart enhancement—especially useful for LLM processing (e.g., ensuring chunks fit within context windows).

We’ll:
1. Use `tiktoken` to count tokens (defaulting to `cl100k_base`, used by GPT-3.5/4).
2. Add a `"token_count"` field to each chunk’s metadata.
3. Show token counts in the chunk preview.
4. Include them in the saved JSON.

---

### 🔧 Step 1: Install `tiktoken`

If not already installed:

```bash
pip install tiktoken
```

---

### ✅ FINAL Code with Token Count (KPI)


```python
import os
import json
from pathlib import Path
from docling.document_converter import DocumentConverter
import gradio as gr
import tiktoken

# Constants
CHUNK_DELIMITER = "<!-- CHUNK -->"
OUTPUT_DIR = Path.cwd()
ENCODING_NAME = "cl100k_base"  # Standard for GPT-3.5/4; change if needed

# Initialize tokenizer once
tokenizer = tiktoken.get_encoding(ENCODING_NAME)

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

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

# Launch
if __name__ == "__main__":
    demo.launch()
```

---

### 📝 Example JSON Output (with token count)

```json
[
  {
    "filename": "paper.md",
    "chunk_index": 0,
    "token_count": 312,
    "text": "Introduction...\n..."
  },
  {
    "filename": "paper.md",
    "chunk_index": 1,
    "token_count": 287,
    "text": "Related Work...\n..."
  }
]
```

---

### 🖼️ UI Preview Example

In the Gradio output textbox, you’ll see:

```
✅ Total chunks: 3 | Total tokens: 842

[Chunk 0] (312 tokens)
Introduction to large language models...

============================================================
[Chunk 1] (287 tokens)
Prior work in document understanding...

...
```

---

### 💡 Notes

- **Encoding**: `cl100k_base` works for `gpt-3.5-turbo`, `gpt-4`, etc.  
  If you’re using another model (e.g., `text-embedding-ada-002`), it also uses `cl100k_base`.
- **Performance**: Tokenization is fast, but for huge files, consider async or progress tracking (not needed for typical use).
- **Empty chunks**: We skip empty/space-only chunks automatically.

---

### ✅ Final Integration

Just replace:
- the `import` section (add `tiktoken`),
- define `tokenizer` and `count_tokens`,
- and use the new `process_chunks`.

Everything else in your Gradio app remains unchanged.

---

