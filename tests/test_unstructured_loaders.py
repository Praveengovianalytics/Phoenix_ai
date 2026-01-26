import pytest

from phoenix_ai.loaders import load_and_process_single_document, load_documents_to_dataframe


@pytest.mark.parametrize("filename,content", [("sample.txt", "Hello world")])
def test_unstructured_single_document(tmp_path, filename, content):
    pytest.importorskip("unstructured")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / filename
    file_path.write_text(content, encoding="utf-8")

    df = load_and_process_single_document(
        folder_path=str(data_dir),
        filename=filename,
        use_unstructured=True,
        chunking_strategy="basic",
    )

    assert not df.empty
    assert "content" in df.columns
    assert "element_type" in df.columns


def test_unstructured_documents_to_dataframe(tmp_path):
    pytest.importorskip("unstructured")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "alpha.txt").write_text("Alpha content", encoding="utf-8")
    (data_dir / "beta.txt").write_text("Beta content", encoding="utf-8")

    df = load_documents_to_dataframe(
        folder_path=str(data_dir),
        use_unstructured=True,
        unstructured_metadata_fields=["page_number", "filetype"],
        chunking_strategy="basic",
    )

    assert not df.empty
    assert "content" in df.columns
    assert "element_type" in df.columns
    assert "filetype" in df.columns
