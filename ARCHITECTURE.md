<p align="center">
  <img src="assets/phoenix_logo.jpeg" alt="Phoenix AI" width="280" />
</p>

# Phoenix AI - Architecture & Design

> Comprehensive technical design document for the Phoenix AI library

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Module Structure](#2-module-structure)
3. [Core Clients Design](#3-core-clients-design)
4. [Document Processing Pipeline](#4-document-processing-pipeline)
5. [Vector Embedding & Storage](#5-vector-embedding--storage)
6. [RAG Inference Engine](#6-rag-inference-engine)
7. [Self-RAG Pipeline](#7-self-rag-pipeline)
8. [GraphRAG Module](#8-graphrag-module)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Agentic Tool System](#10-agentic-tool-system)
11. [Class Hierarchy](#11-class-hierarchy)
12. [Component Interaction Diagram](#12-component-interaction-diagram)
13. [Feature & Provider Matrices](#13-feature--provider-matrices)
14. [Design Principles](#14-design-principles)

---

## 1. System Architecture Overview

Phoenix AI follows a **layered architecture** with six distinct tiers. Each layer depends only on the layers below it, ensuring clean separation of concerns and testability.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PHOENIX AI SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      LAYER 1: PROVIDER ABSTRACTION                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │   │
│  │  │  OpenAI  │ │  Azure   │ │Databricks│ │  Ollama  │ │ HuggingFace  │   │   │
│  │  │   API    │ │ OpenAI   │ │  Model   │ │  Local   │ │   Router     │   │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘   │   │
│  └───────┼────────────┼────────────┼────────────┼──────────────┼───────────┘   │
│          └────────────┴────────────┴────────────┴──────────────┘               │
│                                    │                                            │
│  ┌─────────────────────────────────▼───────────────────────────────────────┐   │
│  │                      LAYER 2: CORE CLIENTS                              │   │
│  │  ┌─────────────────────────┐    ┌─────────────────────────┐             │   │
│  │  │  GenAIEmbeddingClient   │    │    GenAIChatClient      │             │   │
│  │  │  • generate_embedding() │    │    • chat()             │             │   │
│  │  │  • batch processing     │    │    • tool calling       │             │   │
│  │  │  • rate limit handling  │    │    • streaming          │             │   │
│  │  └───────────┬─────────────┘    └───────────┬─────────────┘             │   │
│  └──────────────┼──────────────────────────────┼───────────────────────────┘   │
│                 │                              │                                │
│  ┌──────────────▼──────────────────────────────▼───────────────────────────┐   │
│  │                      LAYER 3: PROCESSING                                │   │
│  │                                                                          │   │
│  │  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────────┐     │   │
│  │  │   Document   │  │ Vector Embedding │  │    Knowledge Graph     │     │   │
│  │  │   Loaders    │  │    Pipeline      │  │      Builder           │     │   │
│  │  │              │  │                  │  │                        │     │   │
│  │  │ • PDF        │  │ • Chunking       │  │ • Entity Extraction    │     │   │
│  │  │ • DOCX       │  │ • Embedding      │  │ • Relationship Extract │     │   │
│  │  │ • TXT/CSV    │  │ • Indexing       │  │ • Graph Assembly       │     │   │
│  │  │ • Images     │  │                  │  │ • Deduplication        │     │   │
│  │  └──────┬───────┘  └────────┬─────────┘  └───────────┬────────────┘     │   │
│  └─────────┼───────────────────┼────────────────────────┼──────────────────┘   │
│            │                   │                        │                       │
│  ┌─────────▼───────────────────▼────────────────────────▼──────────────────┐   │
│  │                      LAYER 4: STORAGE                                    │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐    │   │
│  │  │    FAISS    │ │Azure AI     │ │   Milvus    │ │  Knowledge      │    │   │
│  │  │   (Local)   │ │Search       │ │  (Cloud)    │ │  Graph Store    │    │   │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └────────┬────────┘    │   │
│  └─────────┼───────────────┼───────────────┼─────────────────┼─────────────┘   │
│            └───────────────┴───────────────┴─────────────────┘                  │
│                                    │                                            │
│  ┌─────────────────────────────────▼───────────────────────────────────────┐   │
│  │                      LAYER 5: INFERENCE                                  │   │
│  │  ┌────────────────┐ ┌────────────────┐ ┌────────────────────────────┐   │   │
│  │  │  RAGInferencer │ │SelfRAGInferenc │ │   GraphRAGInferencer       │   │   │
│  │  │                │ │     er         │ │                            │   │   │
│  │  │ • Standard     │ │ • Draft        │ │ • Hybrid Retrieval         │   │   │
│  │  │ • Hybrid       │ │ • Critique     │ │ • Graph Traversal          │   │   │
│  │  │ • HyDE         │ │ • Revise       │ │ • Multi-hop Reasoning      │   │   │
│  │  └────────────────┘ └────────────────┘ └────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│  ┌─────────────────────────────────▼───────────────────────────────────────┐   │
│  │                      LAYER 6: EVALUATION & TOOLS                        │   │
│  │  ┌────────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │   │
│  │  │    RagEvaluator    │  │  Ground Truth   │  │   Agentic Tools     │   │   │
│  │  │                    │  │   Generator     │  │                     │   │   │
│  │  │ • BLEU Score       │  │                 │  │ • Tool Definition   │   │   │
│  │  │ • LLM-as-Judge     │  │ • QA Pairs      │  │ • OpenAI Adapter    │   │   │
│  │  │ • MLflow Logging   │  │ • Diversity     │  │ • JSON Adapter      │   │   │
│  │  └────────────────────┘  └─────────────────┘  └─────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Structure

```
phoenix_ai/
│
├── Core Clients
│   ├── utils.py                    # GenAIEmbeddingClient, GenAIChatClient
│   ├── hf_local_client.py          # HuggingFace local transformer support
│   └── tools.py                    # Agentic tool framework & adapters
│
├── Document Processing
│   └── loaders.py                  # Multi-format document loaders
│
├── Vector Pipeline
│   ├── vector_embedding_pipeline.py # Chunking, embedding, indexing
│   ├── azure_ai_search.py          # Azure AI Search vector store
│   └── milvus_vector_store.py      # Milvus vector store
│
├── RAG Inference
│   ├── rag_inference.py            # RAGInferencer, SelfRAGInferencer
│   ├── self_rag.py                 # Standalone Self-RAG implementation
│   └── config_param.py             # Prompts and configuration constants
│
├── GraphRAG Module
│   ├── __init__.py                 # Module exports
│   ├── knowledge_graph.py          # Entity, Relationship, KnowledgeGraph
│   ├── entity_extractor.py         # LLM, Regex & Hybrid entity extraction
│   ├── relationship_extractor.py   # LLM & co-occurrence relationship extraction
│   ├── graph_builder.py            # Graph construction & incremental builder
│   ├── graph_retriever.py          # Vector + graph hybrid retrieval
│   └── graphrag_inferencer.py      # GraphRAG inference engine
│
├── Evaluation
│   ├── rag_eval.py                 # RagEvaluator with multi-metric scoring
│   ├── eval_dataset_prep_ground_truth.py  # Ground truth QA generation
│   └── rag_evaluation_data_prep.py        # Evaluation data preparation
│
├── Runners
│   └── RAG_Inference_Runner.py     # CLI runner utility
│
└── __init__.py                     # Package exports
```

### Module Dependency Graph

```
                    ┌──────────────────────────┐
                    │     config_param.py       │
                    │   (Prompts & Constants)   │
                    └────────────┬──────────────┘
                                 │
           ┌─────────────────────┼──────────────────────┐
           │                     │                       │
           ▼                     ▼                       ▼
┌─────────────────────┐ ┌──────────────┐ ┌──────────────────────────┐
│    utils.py         │ │  loaders.py  │ │    tools.py              │
│ GenAIEmbeddingClient│ │  Document    │ │  Tool, ToolCall          │
│ GenAIChatClient     │ │  Loading     │ │  ChatAdapter (Protocol)  │
│ hf_local_client.py  │ │              │ │  OpenAI/JSON Adapters    │
└────────┬────────────┘ └──────┬───────┘ └──────────────────────────┘
         │                     │
         ├─────────────────────┤
         │                     │
         ▼                     ▼
┌─────────────────────────────────────────────────┐
│          vector_embedding_pipeline.py            │
│          ┌──────────────────┐                   │
│          │ azure_ai_search  │                   │
│          │ milvus_vector    │                   │
│          └──────────────────┘                   │
└──────────────────────┬──────────────────────────┘
                       │
         ┌─────────────┼─────────────────────────────────┐
         │             │                                  │
         ▼             ▼                                  ▼
┌──────────────┐ ┌──────────────┐          ┌──────────────────────────┐
│rag_inference │ │  self_rag    │          │  graphrag/               │
│              │ │              │          │  ├── knowledge_graph     │
│RAGInferencer │ │SelfRAG       │          │  ├── entity_extractor   │
│              │ │Inferencer    │          │  ├── relationship_ext   │
└──────┬───────┘ └──────────────┘          │  ├── graph_builder      │
       │                                   │  ├── graph_retriever    │
       │                                   │  └── graphrag_inferencer│
       │                                   └─────────────┬────────────┘
       │                                                 │
       │        (inherits RAGInferencer)                │
       └─────────────────────────────────────────────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │  Evaluation Layer     │
           │  ├── rag_eval.py      │
           │  └── eval_dataset_    │
           │      prep_ground_     │
           │      truth.py         │
           └───────────────────────┘
```

---

## 3. Core Clients Design

The two core client classes provide a **unified interface** across all supported LLM/embedding providers. Every other module depends on these clients.

```
┌─────────────────────────────────────────────────────────────────┐
│                    GenAIEmbeddingClient                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  __init__(provider, model, api_key, **kwargs)                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                 Provider Router                             │  │
│  │                                                            │  │
│  │  provider == "openai"              → OpenAI()              │  │
│  │  provider == "azure-openai"        → AzureOpenAI()         │  │
│  │  provider == "databricks"          → OpenAI(base_url=...)  │  │
│  │  provider == "ollama"              → OpenAI(base_url=...)  │  │
│  │  provider == "huggingface"         → OpenAI(base_url=...)  │  │
│  │  provider == "sentence-transformer"→ SentenceTransformer() │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  generate_embedding(texts, batch_size=16, max_retries=3)        │
│  ├── Batched processing for large inputs                        │
│  ├── Exponential backoff on rate limits                         │
│  └── Returns: List[List[float]] | np.ndarray                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      GenAIChatClient                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  __init__(provider, model, api_key, **kwargs)                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                 Provider Router                             │  │
│  │                                                            │  │
│  │  provider == "openai"           → OpenAI()                 │  │
│  │  provider == "azure-openai"     → AzureOpenAI()            │  │
│  │  provider == "databricks"       → OpenAI(base_url=...)     │  │
│  │  provider == "ollama"           → OpenAI(base_url=...)     │  │
│  │  provider == "huggingface"                                 │  │
│  │    ├── use_local_transformer    → HFTextGenClient(local)   │  │
│  │    └── router mode              → OpenAI(base_url=HF)      │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  chat(user_input, system_prompt, max_tokens, temperature)       │
│  ├── Formats messages: [{role: system}, {role: user}]           │
│  ├── Calls provider-specific API                                │
│  └── Returns: str (response content)                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Design Pattern**: Factory pattern via `provider` parameter. All providers expose the same `generate_embedding()` / `chat()` interface, enabling seamless provider swapping.

---

## 4. Document Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DOCUMENT LOADING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT                                                                       │
│  ┌────────────────────────────────────────────────────┐                     │
│  │  Supported Formats:                                │                     │
│  │  .pdf  .docx  .pptx  .xlsx  .csv  .txt  .png .jpg │                     │
│  └───────────────────────┬────────────────────────────┘                     │
│                          │                                                   │
│                          ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                    PARSER SELECTION                           │           │
│  │                                                              │           │
│  │   use_unstructured=False          use_unstructured=True      │           │
│  │   ┌──────────────────┐            ┌──────────────────────┐   │           │
│  │   │  Standard Parsers│            │ Unstructured Library │   │           │
│  │   │                  │            │                      │   │           │
│  │   │  .pdf → PyPDF2   │            │ • hi_res strategy    │   │           │
│  │   │  .docx → docx    │            │ • OCR support        │   │           │
│  │   │  .xlsx → openpyxl│            │ • Table extraction   │   │           │
│  │   │  .csv → pandas   │            │ • Image extraction   │   │           │
│  │   │  .txt → read()   │            │ • Element metadata   │   │           │
│  │   └────────┬─────────┘            └──────────┬───────────┘   │           │
│  └────────────┼─────────────────────────────────┼───────────────┘           │
│               └──────────────┬──────────────────┘                            │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                    CHUNKING ENGINE                            │           │
│  │                                                              │           │
│  │  ┌─────────────────┐  ┌─────────────────┐                   │           │
│  │  │  Simple Split   │  │  Structured     │                   │           │
│  │  │                 │  │  (Unstructured) │                   │           │
│  │  │  chunk_size     │  │                 │                   │           │
│  │  │  overlap        │  │  by_title       │                   │           │
│  │  │  _split_text()  │  │  basic          │                   │           │
│  │  │                 │  │  max_characters  │                   │           │
│  │  └────────┬────────┘  └────────┬────────┘                   │           │
│  └───────────┼────────────────────┼─────────────────────────────┘           │
│              └────────────────────┘                                          │
│                          │                                                   │
│                          ▼                                                   │
│  OUTPUT                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │  pandas DataFrame                                            │           │
│  │  ┌──────────┬────────────────┬──────────┬───────────────┐    │           │
│  │  │ filename │    content     │ chunk_id │   metadata    │    │           │
│  │  ├──────────┼────────────────┼──────────┼───────────────┤    │           │
│  │  │ doc.pdf  │ "Chapter 1..." │ chunk_0  │ {page: 1}    │    │           │
│  │  │ doc.pdf  │ "Section 2..." │ chunk_1  │ {page: 2}    │    │           │
│  │  └──────────┴────────────────┴──────────┴───────────────┘    │           │
│  └──────────────────────────────────────────────────────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Vector Embedding & Storage

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     VECTOR EMBEDDING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────┐                     │
│  │              VectorEmbedding                        │                     │
│  │  __init__(embedding_client, chunk_size, overlap)   │                     │
│  └───────────────────────┬────────────────────────────┘                     │
│                          │                                                   │
│                          ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                   generate_index()                            │           │
│  │                                                              │           │
│  │  1. Read text from DataFrame column                          │           │
│  │  2. Split into chunks (chunk_size + overlap)                 │           │
│  │  3. Generate embeddings via GenAIEmbeddingClient             │           │
│  │  4. Route to appropriate vector store:                       │           │
│  │                                                              │           │
│  │  ┌────────────────────────────────────────────────────────┐  │           │
│  │  │                 INDEX TYPE ROUTER                       │  │           │
│  │  │                                                        │  │           │
│  │  │  "local_index"                                         │  │           │
│  │  │  ┌────────────────────────────────────────────┐       │  │           │
│  │  │  │  FAISS IndexFlatL2                          │       │  │           │
│  │  │  │  • Exact L2 distance search                 │       │  │           │
│  │  │  │  • Saves: .index file + .pkl chunks         │       │  │           │
│  │  │  │  • Best for: <1M vectors, local dev         │       │  │           │
│  │  │  └────────────────────────────────────────────┘       │  │           │
│  │  │                                                        │  │           │
│  │  │  "azure_ai_search_vector_index"                        │  │           │
│  │  │  ┌────────────────────────────────────────────┐       │  │           │
│  │  │  │  AzureAISearchVectorStore                   │       │  │           │
│  │  │  │  • HNSW algorithm (cosine similarity)       │       │  │           │
│  │  │  │  • Configurable: m, ef_construction, ef     │       │  │           │
│  │  │  │  • Supports: API key + Entra ID auth        │       │  │           │
│  │  │  │  • Best for: Enterprise Azure deployments   │       │  │           │
│  │  │  └────────────────────────────────────────────┘       │  │           │
│  │  │                                                        │  │           │
│  │  │  "milvus_vector_index"                                 │  │           │
│  │  │  ┌────────────────────────────────────────────┐       │  │           │
│  │  │  │  MilvusVectorStore                          │       │  │           │
│  │  │  │  • HNSW index (cosine metric)               │       │  │           │
│  │  │  │  • Supports: local + Zilliz cloud           │       │  │           │
│  │  │  │  • Token-based auth for cloud               │       │  │           │
│  │  │  │  • Best for: Scalable cloud deployments     │       │  │           │
│  │  │  └────────────────────────────────────────────┘       │  │           │
│  │  │                                                        │  │           │
│  │  │  "databricks_vector_index"                             │  │           │
│  │  │  ┌────────────────────────────────────────────┐       │  │           │
│  │  │  │  Databricks VectorSearchClient              │       │  │           │
│  │  │  │  • Unity Catalog integration                │       │  │           │
│  │  │  │  • Delta Sync index                         │       │  │           │
│  │  │  │  • Best for: Databricks platform users      │       │  │           │
│  │  │  └────────────────────────────────────────────┘       │  │           │
│  │  └────────────────────────────────────────────────────────┘  │           │
│  └──────────────────────────────────────────────────────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. RAG Inference Engine

### Standard RAG Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌─────────────┐
│  Documents  │────▶│   Loaders    │────▶│  DataFrame      │────▶│  Chunking   │
│  (PDF,DOCX) │     │              │     │  [content,id]   │     │  (overlap)  │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────┬──────┘
                                                                        │
                    ┌──────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────┐
│  GenAIEmbeddingClient   │────▶│    Vector Index         │────▶│   Storage   │
│  • text-embedding-3     │     │    (FAISS/Azure/Milvus) │     │  (index+pkl)│
└─────────────────────────┘     └─────────────────────────┘     └──────┬──────┘
                                                                        │
┌───────────────────────────────────────────────────────────────────────┘
│
▼                              INFERENCE PHASE
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ User Query  │────▶│ Query Embedding │────▶│  Vector Search  │
│             │     │                 │     │  (top_k docs)   │
└─────────────┘     └─────────────────┘     └────────┬────────┘
                                                     │
                    ┌────────────────────────────────┘
                    ▼
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────┐
│   Context Building      │────▶│   GenAIChatClient       │────▶│   Answer    │
│   (retrieved docs)      │     │   (system + context)    │     │  DataFrame  │
└─────────────────────────┘     └─────────────────────────┘     └─────────────┘
```

### RAG Retrieval Modes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAG RETRIEVAL MODES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MODE: "standard"                                                            │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │  Query → Embed → Vector Search (cosine/L2) → Top-K Chunks    │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  MODE: "hybrid"                                                              │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │  Query ──┬──▶ Vector Search ────┐                              │         │
│  │          │                      ├──▶ Score Fusion → Top-K     │         │
│  │          └──▶ Keyword Search ───┘                              │         │
│  │                                                                │         │
│  │  combined = vector_weight × vec_score                         │         │
│  │           + keyword_weight × kw_score                         │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  MODE: "hyde" (Hypothetical Document Embeddings)                             │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │  Query → LLM generates hypothetical answer                    │         │
│  │        → Embed hypothetical answer                            │         │
│  │        → Vector Search with hypothetical embedding            │         │
│  │        → Top-K Chunks                                         │         │
│  │                                                                │         │
│  │  Rationale: Hypothetical answer is semantically closer        │         │
│  │  to real answer chunks than the original question.            │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Self-RAG Pipeline

Self-RAG adds a **critique-and-revise** loop on top of standard RAG to reduce hallucinations.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SELF-RAG PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐           │
│  │ User Query  │────▶│ Vector Search   │────▶│ Retrieved Docs  │           │
│  └─────────────┘     └─────────────────┘     └────────┬────────┘           │
│                                                        │                    │
│                       ┌────────────────────────────────┘                    │
│                       ▼                                                     │
│                 ┌─────────────────────────────────────────────┐             │
│                 │            DRAFT PHASE                      │             │
│                 │                                             │             │
│                 │  Input: system_prompt + context + question  │             │
│                 │                                             │             │
│                 │  GenAIChatClient.chat()                     │             │
│                 │  → "Generate answer using the provided      │             │
│                 │     context documents"                      │             │
│                 │                                             │             │
│                 │  Output: draft_answer                       │             │
│                 └─────────────────┬───────────────────────────┘             │
│                                   │                                         │
│                                   ▼                                         │
│                 ┌─────────────────────────────────────────────┐             │
│                 │           CRITIQUE PHASE                    │             │
│                 │                                             │             │
│                 │  Input: critique_prompt + context +         │             │
│                 │         question + draft_answer             │             │
│                 │                                             │             │
│                 │  GenAIChatClient.chat()                     │             │
│                 │  → "Fact-check this answer against the      │             │
│                 │     provided context. Return JSON:"         │             │
│                 │                                             │             │
│                 │  Output: {                                  │             │
│                 │    verdict: "accept" | "revise",            │             │
│                 │    rationale: "...",                        │             │
│                 │    final_answer: "..." (revised if needed)  │             │
│                 │  }                                          │             │
│                 └─────────────────┬───────────────────────────┘             │
│                                   │                                         │
│                      ┌────────────┴────────────┐                           │
│                      │                         │                            │
│                      ▼                         ▼                            │
│               verdict="accept"          verdict="revise"                   │
│               ┌──────────────┐         ┌──────────────────┐               │
│               │ Use draft as │         │ Use revised      │               │
│               │ final answer │         │ final_answer     │               │
│               └──────────────┘         └──────────────────┘               │
│                      │                         │                            │
│                      └────────────┬────────────┘                           │
│                                   ▼                                         │
│                 ┌─────────────────────────────────────────────┐             │
│                 │  DataFrame Output:                          │             │
│                 │  [question, retrieved_docs, draft_answer,   │             │
│                 │   critique, final_answer]                   │             │
│                 └─────────────────────────────────────────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. GraphRAG Module

GraphRAG is the most architecturally complex module. It enhances standard RAG with a **knowledge graph** for multi-hop reasoning and entity-aware retrieval.

### 8.1 Graph Construction Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       GRAPH CONSTRUCTION PIPELINE                            │
│                       (GraphBuilder.build_from_chunks)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STAGE 1: ENTITY EXTRACTION                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                                                                  │       │
│  │  For each chunk:                                                 │       │
│  │    LLMEntityExtractor.extract(chunk_text, chunk_id)             │       │
│  │                                                                  │       │
│  │  LLM Prompt: "Extract entities from this text.                  │       │
│  │    Entity types: PERSON, ORGANIZATION, LOCATION, PRODUCT,       │       │
│  │    TECHNOLOGY, CONCEPT, EVENT, METRIC, DATE                     │       │
│  │    Return JSON: [{name, type, description}]"                    │       │
│  │                                                                  │       │
│  │  Output: List[Entity] with id, name, type, description,        │       │
│  │          source_chunks                                          │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                          │                                                   │
│                          ▼                                                   │
│  STAGE 2: ENTITY DEDUPLICATION                                               │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                                                                  │       │
│  │  _deduplicate_entities(all_entities)                             │       │
│  │                                                                  │       │
│  │  1. Normalize names: lowercase + strip                          │       │
│  │  2. Group entities by normalized name                           │       │
│  │  3. Merge groups: combine descriptions, source_chunks           │       │
│  │  4. Create ID mapping: old_id → canonical_id                    │       │
│  │                                                                  │       │
│  │  Example:                                                        │       │
│  │    "Steve Jobs" (chunk_0) + "steve jobs" (chunk_2)              │       │
│  │    → merged Entity with source_chunks=[chunk_0, chunk_2]        │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                          │                                                   │
│                          ▼                                                   │
│  STAGE 3: ENTITY EMBEDDING                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                                                                  │       │
│  │  _compute_entity_embeddings(deduplicated_entities)              │       │
│  │                                                                  │       │
│  │  Text representation: "{name}: {description}"                   │       │
│  │  GenAIEmbeddingClient.generate_embedding(texts, batch_size=16)  │       │
│  │  Store in Entity.embedding for later similarity search          │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                          │                                                   │
│                          ▼                                                   │
│  STAGE 4: RELATIONSHIP EXTRACTION                                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                                                                  │       │
│  │  For each chunk (where chunk has >= 2 entities):                │       │
│  │    LLMRelationshipExtractor.extract(entities, chunk, chunk_id)  │       │
│  │                                                                  │       │
│  │  LLM Prompt: "Given these entities found in the text,           │       │
│  │    identify relationships between them.                         │       │
│  │    Relation types: works_for, manages, created_by, located_in,  │       │
│  │    part_of, owns, uses, founded, acquired, invested_in          │       │
│  │    Return JSON with confidence scores."                         │       │
│  │                                                                  │       │
│  │  Output: List[Relationship] with source_id, target_id,         │       │
│  │          relation_type, weight, description                     │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                          │                                                   │
│                          ▼                                                   │
│  STAGE 5: POST-DEDUP RELATIONSHIP UPDATE                                     │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                                                                  │       │
│  │  _update_relationship_ids(relationships, id_mapping)            │       │
│  │                                                                  │       │
│  │  • Remap source_id/target_id to canonical entity IDs            │       │
│  │  • Remove self-relationships (source == target after merge)     │       │
│  │  • Remove duplicate relationships                               │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                          │                                                   │
│                          ▼                                                   │
│  STAGE 6: GRAPH ASSEMBLY                                                     │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                                                                  │       │
│  │  KnowledgeGraph()                                               │       │
│  │  ├── add_entity(entity) for each entity                         │       │
│  │  ├── add_relationship(rel) for each relationship                │       │
│  │  └── Auto-builds:                                               │       │
│  │      • _adjacency_out: {entity_id → [outgoing rels]}           │       │
│  │      • _adjacency_in:  {entity_id → [incoming rels]}           │       │
│  │      • _name_to_id:    {normalized_name → entity_id}           │       │
│  │      • _type_index:    {entity_type → [entity_ids]}            │       │
│  │      • _chunk_to_entities: {chunk_id → [entity_ids]}           │       │
│  │                                                                  │       │
│  │  Graph Operations Available:                                    │       │
│  │  ├── get_neighbors(id, direction, relation_types)               │       │
│  │  ├── find_paths(source, target, max_depth) → BFS               │       │
│  │  ├── get_subgraph(entity_ids, depth)                            │       │
│  │  ├── get_community_chunks(entity_ids)                           │       │
│  │  ├── save(filepath) / load(filepath)                            │       │
│  │  └── to_networkx() → NetworkX graph                             │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Knowledge Graph Data Model

```
┌───────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH DATA MODEL                  │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Entity (@dataclass)                                     │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │  id: str              # MD5 hash (name + type)      │ │  │
│  │  │  name: str            # "Steve Jobs"                │ │  │
│  │  │  entity_type: str     # "PERSON"                    │ │  │
│  │  │  description: str     # "Co-founder of Apple Inc."  │ │  │
│  │  │  source_chunks: []    # ["chunk_0", "chunk_2"]      │ │  │
│  │  │  embedding: []        # [0.023, -0.145, ...]        │ │  │
│  │  │  metadata: {}         # {frequency: 5}              │ │  │
│  │  ├─────────────────────────────────────────────────────┤ │  │
│  │  │  merge_with(other)    # Combine duplicate entities  │ │  │
│  │  │  to_dict() / from_dict()  # Serialization           │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                     connected via                                │
│                          │                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Relationship (@dataclass)                               │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │  source_id: str        # Entity ID                  │ │  │
│  │  │  target_id: str        # Entity ID                  │ │  │
│  │  │  relation_type: str    # "founded"                  │ │  │
│  │  │  weight: float         # 0.95 (confidence)          │ │  │
│  │  │  description: str      # "Jobs co-founded Apple"    │ │  │
│  │  │  evidence_chunks: []   # ["chunk_0"]                │ │  │
│  │  │  metadata: {}          # Additional context          │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  EXAMPLE GRAPH:                                                │
│                                                                │
│  (Steve Jobs)──founded──▶(Apple Inc.)                          │
│       │                       │                                │
│       │                  located_in                             │
│       │                       │                                │
│   created_by                  ▼                                │
│       │               (Cupertino, CA)                          │
│       ▼                                                        │
│   (iPhone)──────produced_by──▶(Apple Inc.)                     │
│       │                                                        │
│   related_to                                                   │
│       │                                                        │
│       ▼                                                        │
│   (Smartphone)                                                 │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### 8.3 Hybrid Retrieval Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GRAPHRAG HYBRID RETRIEVAL                                │
│                     (GraphRAGInferencer.infer)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐                                                               │
│  │  Query:  │                                                               │
│  │  "How is │                                                               │
│  │  Steve   │                                                               │
│  │  Jobs    │                                                               │
│  │  connected                                                               │
│  │  to the  │                                                               │
│  │  iPhone?"│                                                               │
│  └────┬─────┘                                                               │
│       │                                                                      │
│       ├──────────────────────────┬──────────────────────────┐               │
│       │                          │                          │                │
│       ▼                          ▼                          ▼                │
│  ┌─────────────────┐    ┌────────────────────┐    ┌─────────────────────┐   │
│  │  VECTOR PATH    │    │   GRAPH PATH       │    │  NAME MATCH PATH   │   │
│  │                 │    │                    │    │                     │   │
│  │ 1. Embed query  │    │ 1. Embed query     │    │ 1. Fuzzy match     │   │
│  │ 2. FAISS search │    │ 2. Cosine sim vs   │    │    query terms     │   │
│  │    (top_k × 2)  │    │    entity embed.   │    │    against entity  │   │
│  │ 3. Score by     │    │ 3. Top entities    │    │    names           │   │
│  │    distance     │    │ 4. EXPAND:         │    │ 2. Return exact    │   │
│  │                 │    │    traverse graph   │    │    and partial     │   │
│  │ Output:         │    │    neighbors up     │    │    matches         │   │
│  │ [(chunk_id,     │    │    to max_depth     │    │                     │   │
│  │   vec_score)]   │    │ 5. Collect source   │    │ Output:             │   │
│  │                 │    │    chunks           │    │ [Entity]             │   │
│  └────────┬────────┘    └─────────┬──────────┘    └──────────┬──────────┘   │
│           │                       │                          │               │
│           └───────────────────────┼──────────────────────────┘               │
│                                   ▼                                          │
│                    ┌──────────────────────────────────────┐                  │
│                    │         SCORE FUSION                  │                  │
│                    │         (HybridRetriever)             │                  │
│                    │                                      │                  │
│                    │  For each chunk_id:                  │                  │
│                    │                                      │                  │
│                    │  combined_score =                    │                  │
│                    │    vector_weight × vec_score +       │                  │
│                    │    (1 - vector_weight) × graph_score │                  │
│                    │                                      │                  │
│                    │  Default: vector=0.7, graph=0.3      │                  │
│                    │  Sort by combined_score DESC         │                  │
│                    │  Return top_k results                │                  │
│                    └──────────────────┬───────────────────┘                  │
│                                       │                                      │
│                                       ▼                                      │
│                    ┌──────────────────────────────────────┐                  │
│                    │    ENHANCED CONTEXT BUILDER          │                  │
│                    │                                      │                  │
│                    │  Part 1: Document Context            │                  │
│                    │  ─────────────────────────           │                  │
│                    │  "Apple Inc. was founded by Steve    │                  │
│                    │   Jobs in 1976..."                   │                  │
│                    │  "Steve Jobs launched the iPhone     │                  │
│                    │   in 2007..."                        │                  │
│                    │                                      │                  │
│                    │  Part 2: Knowledge Graph Context     │                  │
│                    │  ─────────────────────────────       │                  │
│                    │  Entity: Steve Jobs (PERSON)         │                  │
│                    │    → founded: Apple Inc.             │                  │
│                    │    → created: iPhone                 │                  │
│                    │    → worked_at: Apple Inc.           │                  │
│                    │  Entity: iPhone (PRODUCT)            │                  │
│                    │    → produced_by: Apple Inc.         │                  │
│                    │    → related_to: Smartphone          │                  │
│                    └──────────────────┬───────────────────┘                  │
│                                       │                                      │
│                                       ▼                                      │
│                    ┌──────────────────────────────────────┐                  │
│                    │      GenAIChatClient.chat()          │                  │
│                    │                                      │                  │
│                    │  System: RAG system prompt            │                  │
│                    │  User: Context + Graph + Question    │                  │
│                    │                                      │                  │
│                    │  → Answer with entity-aware context  │                  │
│                    └──────────────────────────────────────┘                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.4 Incremental Graph Building

For large document collections, the `IncrementalGraphBuilder` processes documents in batches:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INCREMENTAL GRAPH BUILDING                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │  Total: 1000 chunks, batch_size=10 → 100 batches            │           │
│  └──────────────────────────────────────────────────────────────┘           │
│                                                                              │
│  Batch 1: chunks[0:10]                                                       │
│  ┌──────────────────────────┐   ┌────────────────────┐                     │
│  │ GraphBuilder.build_from_ │──▶│ batch_graph_1      │                     │
│  │ chunks(batch_1)          │   │ entities: 15       │                     │
│  └──────────────────────────┘   │ relationships: 8   │                     │
│                                  └─────────┬──────────┘                     │
│                                            │  merge                          │
│                                            ▼                                 │
│                                  ┌────────────────────┐                     │
│                                  │ main_graph         │                     │
│                                  │ entities: 15       │                     │
│                                  │ rels: 8            │                     │
│                                  └─────────┬──────────┘                     │
│                                            │  checkpoint                     │
│                                            ▼                                 │
│  Batch 2: chunks[10:20]         ┌────────────────────┐                     │
│  ┌──────────────────────────┐   │ batch_graph_2      │                     │
│  │ GraphBuilder.build_from_ │──▶│ entities: 12       │                     │
│  │ chunks(batch_2)          │   │ relationships: 6   │                     │
│  └──────────────────────────┘   └─────────┬──────────┘                     │
│                                            │  merge (dedup by name)          │
│                                            ▼                                 │
│                                  ┌────────────────────┐                     │
│                                  │ main_graph         │                     │
│                                  │ entities: 24       │ (3 deduplicated)    │
│                                  │ rels: 13           │                     │
│                                  └─────────┬──────────┘                     │
│                                            │  checkpoint                     │
│                                            ▼                                 │
│                                           ...                                │
│                                            │                                 │
│                                            ▼                                 │
│  Batch 100: chunks[990:1000]    ┌────────────────────┐                     │
│                                  │ FINAL GRAPH        │                     │
│                                  │ entities: ~500     │                     │
│                                  │ rels: ~300         │                     │
│                                  │ (with dedup)       │                     │
│                                  └────────────────────┘                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Evaluation Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION WORKFLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STEP 1: GROUND TRUTH GENERATION                                             │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                                                                   │      │
│  │  EvalDatasetGroundTruthGenerator                                 │      │
│  │                                                                   │      │
│  │  Input: DataFrame[content]                                       │      │
│  │                                                                   │      │
│  │  For each chunk:                                                 │      │
│  │    GenAIChatClient + GROUND_TRUTH_PROMPT                         │      │
│  │    → Generate diverse QA pairs from document text                │      │
│  │    → Filter for quality and semantic diversity                   │      │
│  │                                                                   │      │
│  │  Output: DataFrame[question, ground_truth]                       │      │
│  └───────────────────────────────────────┬───────────────────────────┘      │
│                                          │                                   │
│                                          ▼                                   │
│  STEP 2: RAG INFERENCE                                                       │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                                                                   │      │
│  │  RAGInferencer / SelfRAGInferencer / GraphRAGInferencer          │      │
│  │  .infer() for each question                                      │      │
│  │                                                                   │      │
│  │  Output: DataFrame[question, ground_truth, answer,               │      │
│  │                     retrieved_docs]                               │      │
│  └───────────────────────────────────────┬───────────────────────────┘      │
│                                          │                                   │
│                                          ▼                                   │
│  STEP 3: EVALUATION                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                                                                   │      │
│  │  RagEvaluator.evaluate()                                         │      │
│  │                                                                   │      │
│  │  For each (question, ground_truth, answer):                      │      │
│  │                                                                   │      │
│  │  ┌─────────────────────┐    ┌─────────────────────────────────┐  │      │
│  │  │  Traditional Metrics│    │  LLM-as-Judge Metrics           │  │      │
│  │  │                     │    │                                 │  │      │
│  │  │  BLEU Score         │    │  GenAIChatClient +              │  │      │
│  │  │  (nltk.bleu_score)  │    │  EVALUATION_PROMPT              │  │      │
│  │  │                     │    │                                 │  │      │
│  │  │  Precision          │    │  → Answer Relevance  (0.0-1.0) │  │      │
│  │  │  Recall             │    │  → Accuracy          (0.0-1.0) │  │      │
│  │  │  F1 Score           │    │  → Completeness      (0.0-1.0) │  │      │
│  │  │                     │    │  → Clarity            (0.0-1.0) │  │      │
│  │  │                     │    │  → Semantic Sim.      (0-100%) │  │      │
│  │  └─────────────────────┘    └─────────────────────────────────┘  │      │
│  │                                                                   │      │
│  │  ┌───────────────────────────────────────────────────────────┐   │      │
│  │  │                    MLflow Integration                      │   │      │
│  │  │                                                           │   │      │
│  │  │  mlflow.log_metrics({                                    │   │      │
│  │  │    "avg_bleu": 0.42,                                     │   │      │
│  │  │    "avg_relevance": 0.87,                                │   │      │
│  │  │    "avg_accuracy": 0.82,                                 │   │      │
│  │  │    "avg_completeness": 0.79,                             │   │      │
│  │  │    "avg_clarity": 0.91,                                  │   │      │
│  │  │    "precision": 0.85,                                    │   │      │
│  │  │    "recall": 0.78,                                       │   │      │
│  │  │    "f1": 0.81,                                           │   │      │
│  │  │  })                                                      │   │      │
│  │  └───────────────────────────────────────────────────────────┘   │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Agentic Tool System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AGENTIC TOOL SYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DATA STRUCTURES                                                             │
│  ┌─────────────────────┐     ┌─────────────────────┐                       │
│  │  Tool (@dataclass)  │     │ ToolCall (@dataclass)│                       │
│  │  • name             │     │  • id                │                       │
│  │  • description      │     │  • name              │                       │
│  │  • parameters (JSON)│     │  • arguments (dict)  │                       │
│  │  • function (Callable)│   └─────────────────────┘                       │
│  └─────────────────────┘                                                    │
│                                                                              │
│  ADAPTERS (Strategy Pattern)                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                                                                  │       │
│  │  ChatAdapter (Protocol)                                         │       │
│  │  └── create_response(messages, tools) → (content, tool_calls)   │       │
│  │                                                                  │       │
│  │  ┌─────────────────────────┐    ┌────────────────────────────┐  │       │
│  │  │  OpenAIStyleAdapter     │    │   JsonFunctionAdapter      │  │       │
│  │  │                         │    │                            │  │       │
│  │  │  Uses native tool_call  │    │  Instructs LLM to output  │  │       │
│  │  │  API from OpenAI-       │    │  JSON with tool calls:    │  │       │
│  │  │  compatible providers   │    │  {"tool_calls": [...]}    │  │       │
│  │  │                         │    │  or                       │  │       │
│  │  │  Supported:             │    │  {"final_answer": "..."}  │  │       │
│  │  │  • OpenAI               │    │                            │  │       │
│  │  │  • Azure OpenAI         │    │  Supported:                │  │       │
│  │  │  • Databricks           │    │  • Any LLM provider       │  │       │
│  │  └─────────────────────────┘    │  • Ollama, HuggingFace    │  │       │
│  │                                  └────────────────────────────┘  │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│  AGENT LOOP                                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                                                                  │       │
│  │  run_agent_loop(adapter, messages, tools, max_iterations=10)    │       │
│  │                                                                  │       │
│  │  ┌──────────────────────────────────────────────────────────┐   │       │
│  │  │                                                          │   │       │
│  │  │  LOOP:                                                   │   │       │
│  │  │  1. adapter.create_response(messages, tools)             │   │       │
│  │  │     → (content, tool_calls)                              │   │       │
│  │  │                                                          │   │       │
│  │  │  2. If tool_calls:                                       │   │       │
│  │  │     For each tool_call:                                  │   │       │
│  │  │       Execute: tool.function(**arguments)                │   │       │
│  │  │       Append result to messages                          │   │       │
│  │  │     Continue loop                                        │   │       │
│  │  │                                                          │   │       │
│  │  │  3. If no tool_calls (final answer):                     │   │       │
│  │  │     Return content                                       │   │       │
│  │  │                                                          │   │       │
│  │  │  4. If max_iterations reached:                           │   │       │
│  │  │     Return last content                                  │   │       │
│  │  │                                                          │   │       │
│  │  └──────────────────────────────────────────────────────────┘   │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Class Hierarchy

### Extraction Classes

```
                              ┌─────────────────────────┐
                              │    ABC / Protocol        │
                              └───────────┬─────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          │                               │                               │
          ▼                               ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
│  EntityExtractor    │       │RelationshipExtractor│       │    ChatAdapter      │
│  (ABC)              │       │  (ABC)              │       │    (Protocol)       │
│                     │       │                     │       │                     │
│  extract(text,      │       │  extract(entities,  │       │  create_response(   │
│    chunk_id)        │       │    text, chunk_id)  │       │    messages, tools) │
│  → List[Entity]     │       │  → List[Relation]   │       │  → (str, [ToolCall])│
└─────────┬───────────┘       └─────────┬───────────┘       └─────────┬───────────┘
          │                             │                             │
    ┌─────┴──────┐               ┌──────┴──────┐               ┌─────┴─────┐
    │            │               │             │               │           │
    ▼            ▼               ▼             ▼               ▼           ▼
┌─────────┐ ┌─────────┐   ┌─────────┐ ┌────────────┐    ┌────────┐ ┌────────────┐
│   LLM   │ │  Regex  │   │   LLM   │ │Cooccurrence│    │OpenAI  │ │   JSON     │
│ Entity  │ │ Entity  │   │Relation │ │ Relation   │    │Style   │ │ Function   │
│ Extract │ │ Extract │   │ Extract │ │  Extract   │    │Adapter │ │ Adapter    │
└─────────┘ └─────────┘   └─────────┘ └────────────┘    └────────┘ └────────────┘
    │            │
    └──────┬─────┘
           ▼
    ┌──────────────┐
    │   Hybrid     │
    │   Entity     │
    │  Extractor   │
    │              │
    │ Combines LLM │
    │ + Regex with │
    │  dedup       │
    └──────────────┘
```

### RAG Inferencer Hierarchy

```
                    ┌──────────────────────────────┐
                    │        RAGInferencer          │
                    │                              │
                    │  infer()                     │
                    │  _get_query_embedding()      │
                    │  _search_faiss_index()       │
                    │  _search_databricks_index()  │
                    │  _search_azure_index()       │
                    │  _search_milvus_index()      │
                    │  _fuse_results()             │
                    │  _build_context()            │
                    │  _load_chunks()              │
                    └──────────────┬───────────────┘
                                   │
                   ┌───────────────┴───────────────┐
                   │                               │
                   ▼                               ▼
    ┌──────────────────────────┐     ┌──────────────────────────┐
    │   SelfRAGInferencer      │     │  GraphRAGInferencer      │
    │                          │     │                          │
    │   _retrieve_documents()  │     │  build_graph()           │
    │   _run_draft()           │     │  build_graph_from_df()   │
    │   _run_critique()        │     │  load_graph()            │
    │   infer()                │     │  save_graph()            │
    │                          │     │  _retrieve_with_graph()  │
    │   Returns:               │     │  _build_enhanced_context │
    │   [question, draft,      │     │  query_graph()           │
    │    critique, final]      │     │  infer()                 │
    └──────────────────────────┘     │                          │
                                     │  Returns:                │
                                     │  [question, answer,      │
                                     │   entities, graph_stats] │
                                     └──────────────────────────┘
```

### Data Classes

```
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│      Entity          │   │    Relationship       │   │       Tool           │
│    (@dataclass)      │   │    (@dataclass)       │   │    (@dataclass)      │
├──────────────────────┤   ├──────────────────────┤   ├──────────────────────┤
│ id: str              │   │ source_id: str        │   │ name: str            │
│ name: str            │   │ target_id: str        │   │ description: str     │
│ entity_type: str     │   │ relation_type: str    │   │ parameters: Dict     │
│ description: str     │   │ weight: float         │   │ function: Callable   │
│ source_chunks: List  │   │ description: str      │   └──────────────────────┘
│ embedding: Optional  │   │ evidence_chunks: List │
│ metadata: Dict       │   │ metadata: Dict        │   ┌──────────────────────┐
├──────────────────────┤   ├──────────────────────┤   │     ToolCall         │
│ add_source_chunk()   │   │ add_evidence()        │   │    (@dataclass)      │
│ merge_with()         │   │ to_dict()             │   ├──────────────────────┤
│ to_dict()            │   │ from_dict()           │   │ id: str              │
│ from_dict()          │   └──────────────────────┘   │ name: str            │
└──────────────────────┘                               │ arguments: Dict      │
                                                       └──────────────────────┘
```

---

## 12. Component Interaction Diagram

Three-tier architecture showing how user requests flow through the system:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                USER                                          │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                                     │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────────────────┐   │
│  │  RAGInferencer  │  │SelfRAGInferencer│  │  GraphRAGInferencer      │   │
│  │                 │  │                 │  │                           │   │
│  │  infer(         │  │  infer(         │  │  infer(                   │   │
│  │    question,    │  │    question,    │  │    question,              │   │
│  │    index_path,  │  │    index_path,  │  │    index_path,            │   │
│  │    mode,        │  │    critique_    │  │    mode="graphrag",       │   │
│  │    top_k)       │  │    prompt)      │  │    use_graph=True)        │   │
│  └────────┬────────┘  └────────┬────────┘  └─────────────┬─────────────┘   │
│           │                    │                          │                  │
└───────────┼────────────────────┼──────────────────────────┼──────────────────┘
            │                    │                          │
            └────────────────────┼──────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SERVICE LAYER                                       │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     GenAIEmbeddingClient                              │  │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────┐ ┌─────────┐  │  │
│  │  │  OpenAI   │ │   Azure   │ │ Databricks│ │  Ollama │ │SentTrans│  │  │
│  │  └───────────┘ └───────────┘ └───────────┘ └─────────┘ └─────────┘  │  │
│  │                      generate_embedding()                             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                       GenAIChatClient                                 │  │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────┐ ┌─────────┐  │  │
│  │  │  OpenAI   │ │   Azure   │ │ Databricks│ │  Ollama │ │   HF    │  │  │
│  │  └───────────┘ └───────────┘ └───────────┘ └─────────┘ └─────────┘  │  │
│  │                           chat()                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────┐  ┌──────────────────────────────────┐  │
│  │       GraphRetriever           │  │        HybridRetriever          │  │
│  │  • find_similar_entities()     │  │  • retrieve()                   │  │
│  │  • expand_with_graph()         │  │  • score fusion                 │  │
│  │  • get_entity_context()        │  │                                 │  │
│  └────────────────────────────────┘  └──────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                        │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐ │
│  │   Vector Index   │  │ Knowledge Graph  │  │     Document Store         │ │
│  │                  │  │                  │  │                            │ │
│  │  • FAISS         │  │  • Entities      │  │  • Chunks (pickle)         │ │
│  │  • Azure Search  │  │  • Relationships │  │  • Metadata                │ │
│  │  • Milvus        │  │  • Adjacency     │  │  • Source mappings         │ │
│  │  • Databricks    │  │  • JSON/Pickle   │  │                            │ │
│  └──────────────────┘  └──────────────────┘  └────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Feature & Provider Matrices

### Feature Comparison: RAG Modes

| Feature | RAGInferencer | SelfRAGInferencer | GraphRAGInferencer |
|---------|:---:|:---:|:---:|
| Standard Vector Search | ✅ | ✅ | ✅ |
| Keyword Search (Hybrid) | ✅ | ✅ | ✅ |
| HyDE Mode | ✅ | — | ✅ |
| Self-Critique Loop | — | ✅ | — |
| Knowledge Graph | — | — | ✅ |
| Multi-hop Reasoning | — | — | ✅ |
| Entity Extraction | — | — | ✅ |
| Graph Traversal | — | — | ✅ |
| Fact Grounding | — | ✅ | ✅ |
| FAISS Index | ✅ | ✅ | ✅ |
| Azure AI Search | ✅ | ✅ | ✅ |
| Milvus | ✅ | ✅ | ✅ |
| Databricks | ✅ | ✅ | ✅ |

### Supported Providers

| Provider | Embeddings | Chat | Local | Cloud |
|----------|:---:|:---:|:---:|:---:|
| **OpenAI** | ✅ text-embedding-3 | ✅ GPT-4/4o | — | ✅ |
| **Azure OpenAI** | ✅ | ✅ | — | ✅ |
| **Databricks** | ✅ | ✅ | — | ✅ |
| **Ollama** | ✅ | ✅ | ✅ | — |
| **HuggingFace** | ✅ | ✅ Router | ✅ Transformers | ✅ |
| **Sentence Transformers** | ✅ | — | ✅ | — |

### Vector Store Comparison

| Feature | FAISS (Local) | Azure AI Search | Milvus | Databricks |
|---------|:---:|:---:|:---:|:---:|
| Search Algorithm | Flat L2 | HNSW | HNSW | HNSW |
| Similarity Metric | L2 Distance | Cosine | Cosine | Cosine |
| Cloud Native | — | ✅ | ✅ | ✅ |
| Local Development | ✅ | — | ✅ | — |
| Authentication | None | API Key / Entra ID | Token | Workspace |
| Scale | <1M vectors | Enterprise | Enterprise | Enterprise |
| Persistence | File (.index) | Managed | Managed | Delta Lake |

---

## 14. Design Principles

### Provider-Agnostic Architecture

All provider-specific logic is encapsulated in the two core clients (`GenAIEmbeddingClient`, `GenAIChatClient`). Higher layers never interact with provider APIs directly, enabling:

- **Swap providers** without changing application code
- **Test locally** (Ollama/Sentence Transformers) and **deploy to cloud** (OpenAI/Azure)
- **Add new providers** by extending the factory router only

### Composition Over Inheritance

The GraphRAG module demonstrates this principle:
- `GraphBuilder` **composes** an `EntityExtractor` + `RelationshipExtractor`
- `GraphRAGInferencer` **inherits** `RAGInferencer` for vector capabilities, **composes** `GraphRetriever` for graph capabilities
- `HybridRetriever` **composes** a `GraphRetriever` and fuses its scores with vector scores

### DataFrame-Centric Interface

All inferencers return `pandas.DataFrame`, providing:
- Consistent output format across all modes
- Easy integration with pandas-based data pipelines
- Compatibility with MLflow logging and evaluation frameworks

### Incremental Processing

Large document sets are handled via:
- `IncrementalGraphBuilder` for batched graph construction with checkpoints
- Batched embedding generation with rate limit handling
- `add_documents_to_graph()` for appending to existing graphs

### Separation of Concerns

Each module has a single responsibility:
- **Loaders**: Parse documents into DataFrames
- **Vector Pipeline**: Chunk, embed, and index
- **Knowledge Graph**: Data model and graph operations
- **Extractors**: Entity/relationship extraction
- **Retrievers**: Search and scoring
- **Inferencers**: Orchestrate the full RAG pipeline
- **Evaluators**: Measure quality

---

*This document describes Phoenix AI v0.2.21.0. For usage examples, see [README.md](README.md).*
