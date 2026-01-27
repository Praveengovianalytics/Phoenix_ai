<p align="center">
  <img src="assets/phoenix_logo.jpeg" alt="Phoenix AI" width="280" />
</p>

# 🔥 Phoenix_ai

**From prototype to production in GenAI workflows 🚀**

A modular Python library for ML Engineers 🧑‍💻, AI Engineers 🤖, and Software Engineers ⚙️ to build, evaluate, and scale retrieval-augmented generation (RAG), embeddings, and agentic tools — across OpenAI, Azure, Databricks, Ollama, and more.

> ✨ Provider-agnostic • Evaluation-ready • Agentic by design

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/Praveengovianalytics/Phoenix_ai/actions)
[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/phoenix-ai/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

---

## 📋 Project Overview

Phoenix_ai is an open-source, modular Python library that bridges the gap between research prototypes and enterprise-grade AI applications. Built with production-ready architecture, it provides a unified interface for building, evaluating, and scaling GenAI workflows across multiple providers and deployment environments.

---

## 🚀 Key Capabilities

### 🔎 **Vector Embedding & Search**
- Multi-provider embedding generation (OpenAI, Azure, Databricks, Sentence Transformers)
- FAISS-based vector indexing with configurable chunking strategies
- Local and cloud-based vector storage options
- Optimized similarity search with customizable top-k retrieval

### 📚 **Retrieval-Augmented Generation (RAG)**
- Standard, Hybrid, and HyDE (Hypothetical Document Embedding) modes
- Configurable system prompts and retrieval parameters
- Multi-document context processing
- Real-time inference with streaming support

### 🕸️ **GraphRAG - Knowledge Graph Enhanced RAG**
- Automatic entity and relationship extraction using LLMs
- Knowledge graph construction from documents
- Multi-hop reasoning through graph traversal
- Hybrid retrieval combining vector similarity + graph relationships
- Reduced hallucinations through fact grounding
- Save/load knowledge graphs (JSON and pickle formats)

### 📝 **Ground-Truth QA Generation & Evaluation**
- Automated question-answer pair generation from documents
- BLEU score evaluation for answer quality
- LLM-as-a-Judge evaluation using ChatGPT or Claude
- MLflow integration for experiment tracking and metrics logging

### 🛠️ **Agentic Tooling System**
- Provider-agnostic tool definition and execution
- OpenAI-compatible function calling across all supported providers
- JSON-based planning for providers without native tool support
- Modular tool architecture for easy extension and customization

### 🌐 **Provider-Agnostic Design**
- Unified API across OpenAI, Azure OpenAI, Databricks, and Ollama
- Seamless switching between local and cloud deployments
- Consistent interface regardless of underlying provider
- Future-proof architecture for new provider integrations

---

## 🔄 Typical Workflow

1. **Install & Configure** - Set up Phoenix_ai with your preferred providers and authentication
2. **Load & Process Docs** - Import and chunk your documents for vector processing
3. **Embed & Index** - Generate embeddings and create searchable vector indices
4. **RAG Inference** - Perform retrieval-augmented generation with your indexed documents
5. **QA Generation & Eval** - Create evaluation datasets and assess RAG performance
6. **Agentic Tools** - Build and deploy intelligent agents with custom tooling

---

## 💡 Why Phoenix_ai?

### 🧑‍💻 **For ML Engineers**
- Production-ready evaluation frameworks with built-in metrics
- Seamless integration with MLflow for experiment tracking
- Configurable hyperparameters for embedding and retrieval optimization
- Automated ground-truth generation for model validation

### 🤖 **For AI Engineers**
- Rapid prototyping with multiple RAG strategies
- Provider flexibility to test across different LLM ecosystems
- Built-in evaluation pipelines for model comparison
- Agentic tooling for complex reasoning tasks

### ⚙️ **For Software Engineers**
- Modular architecture for easy integration into existing systems
- Consistent APIs across different providers and deployment models
- Comprehensive error handling and logging
- Scalable design patterns for enterprise applications

---

## 🎯 Summary

Phoenix_ai empowers engineers to move from raw ideas → working prototypes → enterprise-grade AI systems. With modular design, provider flexibility, and built-in evaluation, it's the one-stop library for reliable, scalable, and auditable GenAI workflows.

Whether you're building your first RAG application or scaling AI systems for millions of users, Phoenix_ai provides the tools, evaluation frameworks, and architectural patterns you need to succeed in the rapidly evolving GenAI landscape.

---

## 🏗️ Architecture & Design

For comprehensive architecture documentation including system diagrams, data flow diagrams, class hierarchies, and design principles, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

**Quick overview:**

```
phoenix_ai/
├── Core Clients        # GenAIEmbeddingClient, GenAIChatClient (provider-agnostic)
├── Document Processing # Multi-format loaders (PDF, DOCX, XLSX, CSV, TXT, images)
├── Vector Pipeline     # Chunking, embedding, indexing (FAISS, Azure, Milvus)
├── RAG Inference       # Standard, Hybrid, HyDE modes
├── Self-RAG            # Draft → Critique → Revise pipeline
├── GraphRAG            # Knowledge graph construction + hybrid retrieval
├── Evaluation          # BLEU, LLM-as-Judge, MLflow integration
└── Agentic Tools       # Provider-agnostic tool calling framework
```

| Feature | RAGInferencer | SelfRAGInferencer | GraphRAGInferencer |
|---------|:---:|:---:|:---:|
| Vector Search | ✅ | ✅ | ✅ |
| HyDE Mode | ✅ | — | ✅ |
| Self-Critique | — | ✅ | — |
| Knowledge Graph | — | — | ✅ |
| Multi-hop Reasoning | — | — | ✅ |

---

## 📦 Installation

### Prerequisites
- Python 3.11+
- Poetry installed (recommended)

### Install Poetry
```bash
# Homebrew (macOS)
brew install poetry

# pipx (recommended for Python CLIs)
brew install pipx
pipx ensurepath
pipx install poetry

# pip (user install)
python3 -m pip install --user poetry
```

### Clone and Install
```bash
git clone https://github.com/Praveengovianalytics/Phoenix_ai.git
cd Phoenix_ai
poetry install
poetry shell  # optional: activate the venv
```

### Alternative: pip + venv
```bash
git clone https://github.com/Praveengovianalytics/Phoenix_ai.git
cd Phoenix_ai
python3 -m venv .venv
source .venv/bin/activate  # fish: source .venv/bin/activate.fish
pip install -e .
```

---

## ⚙️ Quick Start

### 1. Configure Embedding & Chat Clients

```python
from phoenix_ai.utils import GenAIEmbeddingClient, GenAIChatClient
from phoenix_ai.rag_inference import RAGInferencer, SelfRAGInferencer
from phoenix_ai.config_param import Param

# OpenAI
embedding_client = GenAIEmbeddingClient(
    provider="openai",
    model="text-embedding-3-large",
    api_key="your-openai-key"
)

chat_client = GenAIChatClient(
    provider="openai",
    model="gpt-4o",
    api_key="your-openai-key"
)

# RAG inferencer for standard / HyDE flows
rag_inferencer = RAGInferencer(embedding_client, chat_client)
# Self-RAG inferencer adds a self-critique loop
self_rag_inferencer = SelfRAGInferencer(embedding_client, chat_client)
```

#### Hugging Face (Qwen) via OpenAI-compatible endpoint

```python
import os
from phoenix_ai.utils import GenAIChatClient

chat_client = GenAIChatClient(
    provider="huggingface",
    model="aisingapore/Qwen-SEA-LION-v4-32B-IT",
    api_key=os.environ["HF_TOKEN"],  # Hugging Face token
    # base_url defaults to https://router.huggingface.co/v1; override if needed
)

response = chat_client.chat("What is the capital of France?")
print(response)
```

> Note: When using the HF router, use plain model IDs (e.g., `aisingapore/Qwen-SEA-LION-v4-32B-IT`) and avoid provider suffixes.

#### Hugging Face local/GPU (transformers)

```python
from phoenix_ai.utils import GenAIChatClient

chat_client = GenAIChatClient(
    provider="huggingface",
    model="gpt2",                # or your HF repo
    use_local_transformer=True,  # enable transformers pipeline
    device=0,                    # GPU id, or "cpu"
)
print(chat_client.chat("Hello from a local model!"))
```

### 2. Load and Process Documents

```python
from phoenix_ai.loaders import load_and_process_single_document

df = load_and_process_single_document(
    folder_path="data/", 
    filename="policy_doc.pdf"
)
```

#### Complex documents & images (Unstructured)

Phoenix AI can optionally use the Unstructured parser to handle complex PDFs, Office docs, and images while preserving element metadata.

```python
from phoenix_ai.loaders import load_and_process_single_document

df = load_and_process_single_document(
    folder_path="data/",
    filename="complex_report.pdf",
    use_unstructured=True,
    unstructured_kwargs={
        "strategy": "hi_res",
        "infer_table_structure": True,
    },
    chunking_strategy="by_title",
    chunking_kwargs={
        "max_characters": 1500,
        "combine_text_under_n_chars": 200,
    },
)
```

To load an entire folder (including images such as PNG/JPG), enable the same flag:

```python
from phoenix_ai.loaders import load_documents_to_dataframe

df = load_documents_to_dataframe(
    folder_path="data/",
    use_unstructured=True,
    unstructured_kwargs={"strategy": "hi_res"},
    unstructured_metadata_fields=["page_number", "filetype", "languages", "coordinates"],
    chunking_strategy="by_title",
)
```

#### Example: Complex PDF with table extraction

```python
from phoenix_ai.loaders import load_and_process_single_document

df = load_and_process_single_document(
    folder_path="data/",
    filename="financial_report.pdf",
    use_unstructured=True,
    unstructured_kwargs={
        "strategy": "hi_res",
        "infer_table_structure": True,
        "extract_images_in_pdf": True,
    },
    chunking_strategy="by_title",
    chunking_kwargs={
        "max_characters": 1800,
        "new_after_n_chars": 1600,
    },
)
```

#### Example: Image (scanned receipt/invoice)

```python
from phoenix_ai.loaders import load_and_process_single_document

df = load_and_process_single_document(
    folder_path="data/",
    filename="receipt.png",
    use_unstructured=True,
    unstructured_kwargs={
        "strategy": "hi_res",
        "ocr_languages": "eng",
    },
    chunking_strategy="basic",
    chunking_kwargs={"max_characters": 800},
)
```

#### Example: Folder ingestion with selective metadata

```python
from phoenix_ai.loaders import load_documents_to_dataframe

df = load_documents_to_dataframe(
    folder_path="data/",
    use_unstructured=True,
    unstructured_kwargs={"strategy": "fast"},
    unstructured_metadata_fields=["page_number", "filetype"],
    chunking_strategy="basic",
)
```

### 3. Generate Vector Index

```python
from phoenix_ai.vector_embedding_pipeline import VectorEmbedding

vector = VectorEmbedding(embedding_client, chunk_size=500, overlap=50)
index_path, chunks = vector.generate_index(
    df=df,
    text_column="content",
    index_path="output/policy_doc.index",
    vector_index_type="local_index"
)
```

#### Azure AI Search (vector) with MSI

```python
from phoenix_ai.vector_embedding_pipeline import VectorEmbedding

vector = VectorEmbedding(embedding_client, chunk_size=500, overlap=50)
azure_store = vector.generate_index(
    df=df,
    text_column="content",
    index_path="",  # not used for Azure
    vector_index_type="azure_ai_search_vector_index",
    search_service_endpoint="https://<your-search-service>.search.windows.net",
    search_api_key="<your-search-api-key>",  # or provide credential=ClientSecretCredential(...)
    index_name="policy-index",
    embedding_dim=1536,  # match your embedding model
    content_field_name="content",
    vector_field_name="contentVector",
    title_field_name="title",
    vector_search_profile_name="vector-profile",
    hnsw_algorithm_configuration_name="hnsw-config",
    hnsw_metric="cosine",
    hnsw_m=4,
    hnsw_ef_construction=200,
    hnsw_ef_search=300,
    update_index=True,
)

# azure_store can be passed directly to rag_inferencer.infer(..., index_type="azure_ai_search_vector_index", index=azure_store)
```

#### Azure AI Search (vector) with API key + RAG inference

```python
from phoenix_ai.vector_embedding_pipeline import VectorEmbedding
from phoenix_ai.rag_inference import RAGInferencer
from phoenix_ai.config_param import Param

vector = VectorEmbedding(embedding_client, chunk_size=500, overlap=50)
azure_store = vector.generate_index(
    df=df,
    text_column="content",
    index_path="",
    vector_index_type="azure_ai_search_vector_index",
    search_service_endpoint="https://<your-search-service>.search.windows.net",
    search_api_key="<your-search-api-key>",
    index_name="policy-index",
    embedding_dim=1536,
    content_field_name="content",
    vector_field_name="contentVector",
    title_field_name="title",
    vector_search_profile_name="vector-profile",
    hnsw_algorithm_configuration_name="hnsw-config",
    update_index=True,
)

rag_inferencer = RAGInferencer(embedding_client, chat_client)
response_df = rag_inferencer.infer(
    system_prompt=Param.get_rag_prompt(),
    question="Summarize the policy.",
    top_k=3,
    mode="standard",
    index_type="azure_ai_search_vector_index",
    index=azure_store,
)
print(response_df[["question", "answer"]])
```

#### Milvus (local) with RAG inference

```python
from phoenix_ai.vector_embedding_pipeline import VectorEmbedding
from phoenix_ai.rag_inference import RAGInferencer
from phoenix_ai.config_param import Param

vector = VectorEmbedding(embedding_client, chunk_size=500, overlap=50)
milvus_store = vector.generate_index(
    df=df,
    text_column="content",
    index_path="",
    vector_index_type="milvus_vector_index",
    connection_args={"host": "localhost", "port": "19530"},
    collection_name="policy_index",
    embedding_dim=1536,
    content_field_name="content",
    vector_field_name="embedding",
)

rag_inferencer = RAGInferencer(embedding_client, chat_client)
response_df = rag_inferencer.infer(
    system_prompt=Param.get_rag_prompt(),
    question="Summarize the policy.",
    top_k=3,
    mode="standard",
    index_type="milvus_vector_index",
    index=milvus_store,
)
print(response_df[["question", "answer"]])
```

#### Milvus (cloud hosted) with token auth

```python
from phoenix_ai.vector_embedding_pipeline import VectorEmbedding
from phoenix_ai.rag_inference import RAGInferencer
from phoenix_ai.config_param import Param

vector = VectorEmbedding(embedding_client, chunk_size=500, overlap=50)
milvus_store = vector.generate_index(
    df=df,
    text_column="content",
    index_path="",
    vector_index_type="milvus_vector_index",
    connection_args={
        "uri": "https://<your-cluster>.api.gcp-us-west1.zillizcloud.com",
        "token": "<milvus-token>",
        "secure": True,
    },
    collection_name="policy_index",
    embedding_dim=1536,
    content_field_name="content",
    vector_field_name="embedding",
)

rag_inferencer = RAGInferencer(embedding_client, chat_client)
response_df = rag_inferencer.infer(
    system_prompt=Param.get_rag_prompt(),
    question="Summarize the policy.",
    top_k=3,
    mode="standard",
    index_type="milvus_vector_index",
    index=milvus_store,
)
print(response_df[["question", "answer"]])
```

#### Azure AI Search + Azure OpenAI (RAG) with Entra ID

```python
import os
import pandas as pd
from azure.identity import ClientSecretCredential, get_bearer_token_provider

from phoenix_ai.utils import GenAIChatClient, GenAIEmbeddingClient
from phoenix_ai.vector_embedding_pipeline import VectorEmbedding
from phoenix_ai.rag_inference import RAGInferencer
from phoenix_ai.config_param import Param

credential = ClientSecretCredential(
    tenant_id=os.environ["AZURE_TENANT_ID"],
    client_id=os.environ["AZURE_CLIENT_ID"],
    client_secret=os.environ["AZURE_CLIENT_SECRET"],
)
token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)

embedding_client = GenAIEmbeddingClient(
    provider="azure-openai",
    model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_ad_token_provider=token_provider,
)

chat_client = GenAIChatClient(
    provider="azure-openai",
    model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_ad_token_provider=token_provider,
)

df = pd.DataFrame(
    [
        {"title": "Azure AI Search Overview", "content": "Azure AI Search provides full text and vector search."},
        {"title": "Vector Search", "content": "Vector search enables semantic similarity with embeddings."},
        {"title": "RAG Architecture", "content": "RAG combines retrieval with LLMs for grounded answers."},
    ]
)

vector = VectorEmbedding(embedding_client, chunk_size=500, overlap=50)
azure_store = vector.generate_index(
    df=df,
    text_column="content",
    index_path="",
    vector_index_type="azure_ai_search_vector_index",
    search_service_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    search_api_key=os.environ["AZURE_SEARCH_API_KEY"],
    index_name="sample-text-index",
    embedding_dim=1536,
    content_field_name="content",
    vector_field_name="contentVector",
    title_field_name="title",
    vector_search_profile_name="vector-profile",
    hnsw_algorithm_configuration_name="hnsw-config",
    update_index=True,
)

rag_inferencer = RAGInferencer(embedding_client, chat_client)
result_df = rag_inferencer.infer(
    system_prompt=Param.get_rag_prompt(),
    question="What is RAG and how does Azure AI Search help?",
    top_k=3,
    mode="standard",
    index_type="azure_ai_search_vector_index",
    index=azure_store,
)
print(result_df[["question", "answer"]])
```

### 4. Perform RAG Inference (Standard, Hybrid, or HyDE)

```python
# Standard RAG
response_df = rag_inferencer.infer(
    system_prompt=Param.get_rag_prompt(),
    index_path="output/policy_doc.index",
    question="What is the purpose of the company Group Data Classification Policy?",
    mode="standard",
    top_k=5
)

# HyDE RAG (generate hypothetical answer to guide retrieval)
hyde_df = rag_inferencer.infer(
    system_prompt=Param.get_rag_prompt(),
    index_path="output/policy_doc.index",
    question="What data categories are mentioned?",
    mode="hyde",
    top_k=5
)
```

### 5. Generate Evaluation Dataset

```python
from phoenix_ai.eval_dataset_prep_ground_truth import EvalDatasetGroundTruthGenerator

generator = EvalDatasetGroundTruthGenerator(chat_client)
qa_df = generator.process_dataframe(
    df=df,
    text_column="content",
    prompt_template=Param.get_ground_truth_prompt(),
    max_total_pairs=50
)
```

### Testing

See `TESTING.md` for detailed test instructions.

### 6. Evaluate RAG Performance

```python
from phoenix_ai.rag_eval import RagEvaluator

evaluator = RagEvaluator(chat_client, experiment_name="rag_evaluation")
df_eval, metrics = evaluator.evaluate(
    input_df=result_df,
    prompt=Param.get_evaluation_prompt(),
    max_rows=5
)

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
```

### 7. Run Self-RAG (context-aware + self-critique)

```python
# Use the SelfRAGInferencer instance defined above (draft + critique)
self_rag_df = self_rag_inferencer.infer(
    system_prompt=Param.get_rag_prompt(),
    critique_prompt=Param.get_self_rag_critique_prompt(),
    index_path="output/policy_doc.index",
    question="What are the payment terms?",
    index_type="local_index",  # or "databricks_vector_index" with index object
    top_k=3,
    max_tokens=256,
)

# Inspect both the draft and the final self-critiqued answer
print(self_rag_df[["draft_answer", "final_answer"]])
```

### 8. GraphRAG - Knowledge Graph Enhanced RAG

GraphRAG enhances traditional RAG with knowledge graph capabilities for improved multi-hop reasoning and reduced hallucinations.

#### Basic GraphRAG Setup

```python
from phoenix_ai.utils import GenAIEmbeddingClient, GenAIChatClient
from phoenix_ai.graphrag import GraphRAGInferencer

# Initialize clients
embedding_client = GenAIEmbeddingClient(
    provider="openai",
    model="text-embedding-3-small",
    api_key="your-openai-key"
)

chat_client = GenAIChatClient(
    provider="openai",
    model="gpt-4o-mini",
    api_key="your-openai-key"
)

# Create GraphRAG inferencer
graphrag = GraphRAGInferencer(
    embedding_client=embedding_client,
    chat_client=chat_client,
    graph_weight=0.3,        # Balance between vector and graph retrieval
    max_graph_depth=2,       # Multi-hop traversal depth
    include_graph_context=True
)
```

#### Build Knowledge Graph from Documents

```python
# From a list of text chunks
chunks = [
    "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
    "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
    "Steve Jobs returned to Apple in 1997 and launched the iPhone in 2007.",
]

# Build the knowledge graph (extracts entities and relationships)
graph = graphrag.build_graph(chunks)

print(f"Entities: {graph.num_entities}")
print(f"Relationships: {graph.num_relationships}")

# Or build from a DataFrame (compatible with document loaders)
from phoenix_ai.loaders import load_and_process_single_document

df = load_and_process_single_document(folder_path="data/", filename="company_docs.pdf")
graph = graphrag.build_graph_from_dataframe(df, text_column="content")
```

#### Perform GraphRAG Inference

```python
from phoenix_ai.config_param import Param

# First, create the FAISS index for vector retrieval
from phoenix_ai.vector_embedding_pipeline import VectorEmbedding

vector = VectorEmbedding(embedding_client, chunk_size=500, overlap=50)
index_path, _ = vector.generate_index(
    df=df,
    text_column="content",
    index_path="output/company_docs.index",
    vector_index_type="local_index"
)

# Run GraphRAG inference (combines vector + graph retrieval)
result_df = graphrag.infer(
    system_prompt=Param.get_rag_prompt(),
    question="How is Steve Jobs connected to Apple's product innovations?",
    index_path=index_path,
    top_k=5,
    mode="graphrag",     # Use graph-enhanced retrieval
    use_graph=True
)

print(result_df[["question", "answer"]])

# Access extracted entities used in the query
if "entities" in result_df.columns:
    print("Relevant entities:", result_df["entities"].iloc[0])
```

#### Save and Load Knowledge Graphs

```python
# Save the knowledge graph for later use
graphrag.save_graph("output/company_knowledge_graph.json")

# Load a previously built knowledge graph
graphrag.load_graph("output/company_knowledge_graph.json")

# The graph can also be saved in pickle format for faster loading
graph.save("output/company_graph.pkl")
loaded_graph = KnowledgeGraph.load("output/company_graph.pkl")
```

#### Direct Graph Querying (for debugging/exploration)

```python
# Query the knowledge graph directly without LLM generation
results = graphrag.query_graph(
    query="Apple iPhone",
    k=5
)

for r in results["results"]:
    entity = r["entity"]
    print(f"Entity: {entity['name']} ({entity['entity_type']})")
    print(f"  Score: {r['score']:.3f}")
    print(f"  Related chunks: {r['chunks']}")
```

#### Advanced: Custom Entity and Relationship Types

```python
from phoenix_ai.graphrag import GraphBuilder, LLMEntityExtractor, LLMRelationshipExtractor

# Define custom extraction types for your domain
entity_extractor = LLMEntityExtractor(
    chat_client=chat_client,
    entity_types=["DRUG", "DISEASE", "SYMPTOM", "TREATMENT", "GENE"]
)

relationship_extractor = LLMRelationshipExtractor(
    chat_client=chat_client,
    relation_types=["TREATS", "CAUSES", "INHIBITS", "ASSOCIATED_WITH", "TARGETS"]
)

# Build a domain-specific graph
builder = GraphBuilder(
    entity_extractor=entity_extractor,
    relationship_extractor=relationship_extractor,
    embedding_client=embedding_client,
    deduplicate_entities=True
)

medical_graph = builder.build_from_chunks(medical_chunks)
```

#### Incremental Graph Building (for large document sets)

```python
from phoenix_ai.graphrag import IncrementalGraphBuilder, GraphBuilder

# For large document collections, build incrementally
builder = GraphBuilder.from_clients(
    chat_client=chat_client,
    embedding_client=embedding_client
)

incremental_builder = IncrementalGraphBuilder(
    graph_builder=builder,
    batch_size=10,
    checkpoint_path="output/graph_checkpoint.json"
)

# Build with progress tracking
def progress_callback(current, total, stage):
    print(f"{stage}: {current}/{total}")

large_graph = incremental_builder.build(
    chunks=large_chunk_list,
    progress_callback=progress_callback
)
```

---

## 🛠️ Supported Providers

- **🧠 OpenAI** - GPT-4, GPT-3.5, text-embedding models
- **☁️ Azure OpenAI** - Enterprise-grade OpenAI services
- **💼 Databricks** - Model serving and MosaicML integration
- **🤗 Hugging Face (Qwen, etc.)** - Open-source model support via OpenAI-compatible endpoint
- **🏠 Ollama** - Local LLM deployment and inference
- **🔓 Sentence Transformers** - Free local embedding generation

---

## 📚 Documentation

For detailed usage examples, API reference, and advanced configurations, please refer to the project documentation and code examples in the repository.

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
