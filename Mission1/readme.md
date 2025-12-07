# File for the Substack Article RAG that works - Mission #1

Built with Docling, an open-source, high-fidelity document converter from IBM.

Docling simplifies document processing, parsing diverse formats — including advanced PDF understanding — and providing seamless integrations with the gen AI ecosystem.

This tab, is both a standalone app, and an orchestration element. By itself it will:

Accepts a PDF upload

Converts it to clean Markdown (preserving tables, lists, headings)

Saves the output as a .md file in your working directory

No cloud APIs. No hidden costs. 


### Requirements
```bash
pip install "docling[rapidocr]" requests gradio tiktoken
```

To run use
```bash
python docling_1.py
```
