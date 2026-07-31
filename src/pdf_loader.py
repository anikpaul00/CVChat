import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter


def load_pdf(pdf_path: str):
    
    pdf_loader = pymupdf4llm.to_markdown(pdf_path)
    
    headers_to_split_on = [
    ("#", "Section"),
    ("##", "Section"),
    ("###", "SubSection"),
    ("####", "DetailHeader"),   
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False
    )
    
    chunks = markdown_splitter.split_text(pdf_loader)
    
    return chunks
