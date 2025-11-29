import os
import json
from pathlib import Path
from docling.document_converter import DocumentConverter
import gradio as gr

# Constants
OUTPUT_DIR = Path.cwd()

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

# Launch
if __name__ == "__main__":
    demo.launch()
