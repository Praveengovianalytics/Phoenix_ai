import os
import pandas as pd
from typing import List
from PyPDF2 import PdfReader
import tiktoken
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader, UnstructuredExcelLoader
)
def _read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def _split_text(text: str, max_chars: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into chunks based on character count.

    Args:
        text: Input text to split.
        max_chars: Maximum characters per chunk (default: 1000).
        overlap: Number of overlapping characters between chunks (default: 100).

    Returns:
        List of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Ensure we don't split in the middle of a word
        if end < len(text):
            while end > start and text[end] not in ' \n\t':
                end -= 1
            if end == start:
                end = start + max_chars  # Fallback to hard cut if no space found
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap if end - overlap > start else start + max_chars
    return chunks

def load_and_process_single_document(folder_path: str, filename: str) -> pd.DataFrame:
    """
    Reads a document (PDF or .txt), splits it into chunks, and returns a DataFrame.
    
    Returns:
        DataFrame with columns: [filename, page_num, chunk_id, content]
    """
    file_path = os.path.join(folder_path, filename)
    
    if filename.lower().endswith(".pdf"):
        full_text = _read_pdf(file_path)
    elif filename.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    else:
        raise ValueError("Unsupported file type. Only .pdf and .txt are supported.")
    
    chunks = _split_text(full_text)
    
    return pd.DataFrame({
        "filename": [filename] * len(chunks),
        "chunk_id": list(range(len(chunks))),
        "content": chunks
    })

def load_documents_to_dataframe(folder_path: str) -> pd.DataFrame:
    """
    Reads multiple documents (PDF or .txt), splits it into chunks, and returns a DataFrame.
    
    Returns:
        DataFrame with columns: [filename, content]
    """
    supported_loaders = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".xlsx": UnstructuredExcelLoader
    }

    records = []
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[-1].lower()
        loader_class = supported_loaders.get(ext)
        if loader_class:
            print(f" Loading: {filename}")
            loader = loader_class(os.path.join(folder_path, filename))
            documents = loader.load()
            for doc in documents:
                records.append({
                    "filename": filename,
                    "content": doc.page_content.strip()
                })

    return pd.DataFrame(records)
