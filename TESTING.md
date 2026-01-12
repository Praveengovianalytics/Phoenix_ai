# Testing Phoenix_ai

## Overview
The test suite focuses on validating the vector indexing pipeline and Azure AI Search
integration logic without requiring live Azure resources. Azure SDK dependencies are
mocked where necessary so tests can run locally.

## Requirements
* Python 3.11+
* Project dependencies installed (see `pyproject.toml`)

## Run tests
```bash
pytest
```

## Azure AI Search tests
The Azure AI Search tests validate document shaping, field mapping, and pipeline
behavior for Azure-backed indexes by replacing the Azure client with a test double.

Relevant tests:
* `tests/test_vector_embedding_pipeline_azure.py::test_generate_index_azure_ai_search_maps_fields`
