#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["docling>=2.58.0", "requests>=2.32.5"]
# ///

import tempfile
import os
import requests
from pydantic import AnyUrl
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption  # 👈 Import this!

pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"

with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
    f.write(requests.get(pdf_url).content)
    temp_path = f.name

try:
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        lmm_options=ApiVlmOptions(
            url=AnyUrl("http://127.0.0.1:8080/v1/chat/completions"),
            params={"model": "danchev/ibm-granite-docling-258m-GGUF"},
            prompt="You are a document parser. Convert the following structured document content into clean markdown.",
            temperature=0.0,
            response_format=ResponseFormat.MARKDOWN,
            timeout=200,
        ),
    )

    # ✅ Wrap PdfPipelineOptions in PdfFormatOption
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = doc_converter.convert(temp_path)
    print(result.document.export_to_markdown())

finally:
    os.unlink(temp_path)