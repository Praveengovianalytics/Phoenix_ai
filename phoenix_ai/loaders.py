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

def _split_text(text: str, max_tokens: int = 300, overlap: int = 50) -> List[str]:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = encoding.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap

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