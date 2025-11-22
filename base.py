
from docling.document_converter import DocumentConverter
from easygui import fileopenbox
import os

current_directory = os.getcwd()

# Change this to a local path or another URL if desired.
# Note: using the default URL requires network access; if offline, provide a
# local file path (e.g., Path("/path/to/file.pdf")).
source = fileopenbox(msg='Pick your PDF', default='*.pdf')
#source = "https://arxiv.org/pdf/2408.09869"

converter = DocumentConverter()
result = converter.convert(source)

# Print Markdown to stdout.
finalMD = result.document.export_to_markdown()
textfilename = f'{source.split('\\')[-1][:-3]}md'
with open(textfilename, "w", encoding='utf-8') as f:
    f.write(finalMD)
f.close()
print(result.document.export_to_markdown())