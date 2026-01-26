import os
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredExcelLoader,
    UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader)
from PyPDF2 import PdfReader

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
DEFAULT_UNSTRUCTURED_METADATA_FIELDS = (
    "page_number",
    "filetype",
    "source",
    "languages",
    "coordinates",
)


# Ensure the data directory exists
def ensure_folder_exists(folder_path: str):
    try:
        os.makedirs(folder_path, exist_ok=True)
    except Exception as e:
        print(f"❌ Failed to create folder {folder_path}: {e}")
        raise


def _read_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        raise FileNotFoundError(f"❌ Failed to read PDF file {file_path}: {e}")


def _split_text(text: str, max_chars: int = 1000, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            while end > start and text[end] not in " \n\t":
                end -= 1
            if end == start:
                end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap if end - overlap > start else start + max_chars
    return chunks


def _partition_unstructured(
    file_path: str,
    file_extension: str,
    unstructured_kwargs: Optional[Dict[str, object]] = None,
):
    try:
        if file_extension == ".pdf":
            from unstructured.partition.pdf import partition_pdf

            return partition_pdf(filename=file_path, **(unstructured_kwargs or {}))
        if file_extension in IMAGE_EXTENSIONS:
            from unstructured.partition.image import partition_image

            return partition_image(filename=file_path, **(unstructured_kwargs or {}))
        from unstructured.partition.auto import partition

        return partition(filename=file_path, **(unstructured_kwargs or {}))
    except ImportError as exc:
        raise ImportError(
            "Unstructured is required for complex document parsing. "
            "Install it with `pip install unstructured`."
        ) from exc


def _chunk_unstructured_elements(
    elements: Sequence[object],
    chunking_strategy: Optional[str],
    chunking_kwargs: Optional[Dict[str, object]] = None,
) -> Sequence[object]:
    if not chunking_strategy:
        return elements
    try:
        if chunking_strategy == "by_title":
            from unstructured.chunking.title import chunk_by_title

            return chunk_by_title(elements, **(chunking_kwargs or {}))
        if chunking_strategy == "basic":
            from unstructured.chunking.basic import chunk_elements

            return chunk_elements(elements, **(chunking_kwargs or {}))
    except ImportError as exc:
        raise ImportError(
            "Unstructured chunking requires the unstructured package. "
            "Install it with `pip install unstructured`."
        ) from exc
    raise ValueError(
        "Unsupported chunking_strategy. Use 'by_title', 'basic', or None."
    )


def _serialize_metadata_value(value: object) -> object:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return value


def _records_from_unstructured(
    elements: Sequence[object],
    filename: str,
    metadata_fields: Iterable[str],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for element_index, element in enumerate(elements):
        text = (getattr(element, "text", "") or "").strip()
        if not text:
            continue
        metadata = getattr(element, "metadata", None)
        record: Dict[str, object] = {
            "filename": filename,
            "content": text,
            "chunk_id": element_index,
            "element_type": getattr(element, "category", None)
            or element.__class__.__name__,
        }
        if metadata is not None:
            for field in metadata_fields:
                record[field] = _serialize_metadata_value(
                    getattr(metadata, field, None)
                )
        records.append(record)
    return records


def load_and_process_single_document(
    folder_path: str,
    filename: str,
    *,
    use_unstructured: bool = False,
    unstructured_kwargs: Optional[Dict[str, object]] = None,
    unstructured_metadata_fields: Optional[Iterable[str]] = None,
    chunking_strategy: Optional[str] = None,
    chunking_kwargs: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    ensure_folder_exists(folder_path)
    file_path = os.path.join(folder_path, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    ext = os.path.splitext(filename)[-1].lower()
    try:
        if use_unstructured:
            elements = _partition_unstructured(
                file_path, ext, unstructured_kwargs=unstructured_kwargs
            )
            elements = _chunk_unstructured_elements(
                elements, chunking_strategy, chunking_kwargs
            )
            records = _records_from_unstructured(
                elements,
                filename,
                unstructured_metadata_fields
                or DEFAULT_UNSTRUCTURED_METADATA_FIELDS,
            )
            if records:
                return pd.DataFrame(records)

        if filename.lower().endswith(".pdf"):
            full_text = _read_pdf(file_path)
        elif filename.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()
        elif filename.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
            rows_as_text = [
                " | ".join(str(v) for v in row if pd.notna(v))
                for _, row in df.iterrows()
            ]
            full_text = "\n".join(rows_as_text)
        else:
            raise ValueError(
                "❌ Unsupported file type. Only .pdf, .txt, and .csv are supported."
            )
    except Exception as e:
        raise ValueError(f"❌ Error reading {filename}: {e}")

    chunks = _split_text(full_text)
    return pd.DataFrame(
        {
            "filename": [filename] * len(chunks),
            "chunk_id": list(range(len(chunks))),
            "content": chunks,
        }
    )


def load_documents_to_dataframe(
    folder_path: str,
    *,
    use_unstructured: bool = False,
    unstructured_kwargs: Optional[Dict[str, object]] = None,
    unstructured_metadata_fields: Optional[Iterable[str]] = None,
    chunking_strategy: Optional[str] = None,
    chunking_kwargs: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    ensure_folder_exists(folder_path)

    supported_loaders = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".xlsx": UnstructuredExcelLoader,
    }

    records = []
    try:
        filenames = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"❌ Folder not found: {folder_path}")
        return pd.DataFrame()

    for filename in filenames:
        ext = os.path.splitext(filename)[-1].lower()
        file_path = os.path.join(folder_path, filename)

        try:
            if use_unstructured and (
                ext in supported_loaders or ext in IMAGE_EXTENSIONS
            ):
                print(f"🧩 Loading with Unstructured: {filename}")
                elements = _partition_unstructured(
                    file_path, ext, unstructured_kwargs=unstructured_kwargs
                )
                elements = _chunk_unstructured_elements(
                    elements, chunking_strategy, chunking_kwargs
                )
                records.extend(
                    _records_from_unstructured(
                        elements,
                        filename,
                        unstructured_metadata_fields
                        or DEFAULT_UNSTRUCTURED_METADATA_FIELDS,
                    )
                )

            elif ext == ".csv":
                print(f"📄 Loading CSV: {filename}")
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    record_text = " | ".join(
                        str(v) for v in row.values if pd.notna(v)
                    )
                    records.append({"filename": filename, "content": record_text})

            elif ext in [".xls", ".xlsx"]:
                print(f"📊 Loading Excel: {filename}")
                excel_data = pd.read_excel(file_path, sheet_name=None)
                for sheet_name, sheet_df in excel_data.items():
                    content = sheet_df.to_string(index=False)
                    records.append(
                        {"filename": f"{filename}::{sheet_name}", "content": content}
                    )

            elif ext in supported_loaders:
                print(f"📘 Loading with LangChain loader: {filename}")
                loader = supported_loaders[ext](file_path)
                documents = loader.load()
                for doc in documents:
                    records.append(
                        {"filename": filename, "content": doc.page_content.strip()}
                    )

            elif ext == ".txt":
                print(f"📝 Loading TXT: {filename}")
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    records.append({"filename": filename, "content": content})

            else:
                print(f"⏭️ Skipping unsupported file: {filename}")

        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

    return pd.DataFrame(records)
