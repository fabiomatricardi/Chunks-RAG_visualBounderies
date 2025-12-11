# Files for the Substack Article RAG that works - `Mission #2 and #2.1`

It is something like this:
```bash
docling
    ├───RAGworkLIB.py
    └───RAGwork_Mission2.py
```
RAGworkLIB.py will contain all the helper functions, organized by TAB

RAGwork_Mission2.py will have all the GUI logic

<img src='https://github.com/fabiomatricardi/Chunks-RAG_visualBounderies/raw/main/Mission2/TAB2_gif.gif' width=900>

### Requirements
```bash
pip install openai "docling[rapidocr]" requests gradio tiktoken
```

To launch the gradio app, from the terminal
```bash
python RAGwork_Mission2.py
```
