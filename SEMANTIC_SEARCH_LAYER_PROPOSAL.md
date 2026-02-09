# Semantic Search Layer Proposal for Phoenix AI

## Comprehensive Analysis: Unified Graph + Vector + Text Retrieval

> **Author**: Analysis prepared for Phoenix AI enhancement
> **Date**: February 2026
> **Status**: Design Proposal

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [The Case for a Unified Semantic Layer](#3-the-case-for-a-unified-semantic-layer)
4. [Proposed Architecture](#4-proposed-architecture)
5. [Component Deep Dive](#5-component-deep-dive)
6. [Fusion Algorithms](#6-fusion-algorithms)
7. [Query Understanding & Routing](#7-query-understanding--routing)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Comparison: Before vs After](#9-comparison-before-vs-after)
10. [Performance Benchmarks](#10-performance-benchmarks)
11. [References](#11-references)

---

## 1. Executive Summary

### The Problem

Current RAG systems, including Phoenix AI's existing implementation, typically rely on **single-modality retrieval**:
- **Vector RAG**: Semantic similarity via embeddings (misses explicit relationships)
- **GraphRAG**: Relationship traversal (misses semantic nuance in unstructured text)
- **Keyword Search**: Exact matching (misses semantic understanding)

Each approach has blind spots that the others can fill.

### The Solution

A **Unified Semantic Search Layer** that orchestrates all three retrieval modalities:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SEMANTIC SEARCH LAYER                                 │
│                                                                          │
│   ┌───────────────┐   ┌───────────────┐   ┌───────────────────────┐    │
│   │    Vector     │   │    Graph      │   │    Sparse/Lexical     │    │
│   │   Retrieval   │   │   Traversal   │   │      Retrieval        │    │
│   │   (Dense)     │   │  (Structured) │   │   (BM25/SPLADE)       │    │
│   └───────┬───────┘   └───────┬───────┘   └───────────┬───────────┘    │
│           │                   │                       │                 │
│           └───────────────────┼───────────────────────┘                 │
│                               ▼                                          │
│                    ┌─────────────────────┐                              │
│                    │   Score Fusion      │                              │
│                    │   (RRF / Learned)   │                              │
│                    └──────────┬──────────┘                              │
│                               ▼                                          │
│                    ┌─────────────────────┐                              │
│                    │     Re-Ranking      │                              │
│                    │   (Cross-Encoder)   │                              │
│                    └──────────┬──────────┘                              │
│                               ▼                                          │
│                       Unified Results                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Benefits

| Metric | Vector Only | Graph Only | **Unified Layer** |
|--------|-------------|------------|-------------------|
| Semantic Understanding | ✅ High | ⚠️ Limited | ✅ High |
| Relationship Reasoning | ❌ None | ✅ High | ✅ High |
| Exact Match Precision | ❌ Poor | ⚠️ Varies | ✅ High |
| Multi-hop Questions | ❌ Fails | ✅ Strong | ✅ Strong |
| Scalability | ✅ Excellent | ⚠️ Moderate | ✅ Good |
| Explainability | ❌ Black box | ✅ Clear | ✅ Clear |

---

## 2. Current State Analysis

### 2.1 Phoenix AI Current Architecture

Phoenix AI currently offers three inference modes:

#### RAGInferencer (Vector-based)
```python
# Current flow
query → embed → FAISS/Azure/Milvus search → top_k chunks → LLM → answer
```

**Strengths:**
- Fast semantic similarity search
- Handles large document collections
- Simple implementation

**Limitations:**
- Chunks treated as independent units (no relationships)
- Context fragmentation across chunks
- Cannot answer "How is X related to Y?" questions
- Similarity ≠ Relevance for complex queries

#### GraphRAGInferencer (Graph + Vector hybrid)
```python
# Current flow
query → entity extraction → graph traversal + vector search → fusion → LLM → answer
```

**Strengths:**
- Captures entity relationships
- Multi-hop reasoning capability
- Enhanced context from graph structure

**Limitations:**
- No lexical/keyword component
- Graph construction is expensive
- Missing sparse retrieval for exact matches

### 2.2 What's Missing

```
CURRENT PHOENIX GAPS:

 ┌─────────────────────────────────────────────────────────────────┐
 │                                                                  │
 │  1. SPARSE/LEXICAL RETRIEVAL                                    │
 │     • No BM25 implementation                                    │
 │     • No SPLADE sparse embeddings                               │
 │     • Exact term matching is weak                               │
 │                                                                  │
 │  2. QUERY UNDERSTANDING                                          │
 │     • No query decomposition for complex questions              │
 │     • No semantic routing (simple vs complex)                   │
 │     • No entity linking to knowledge graph                      │
 │                                                                  │
 │  3. ADVANCED FUSION                                              │
 │     • Basic weighted averaging only                             │
 │     • No RRF (Reciprocal Rank Fusion)                          │
 │     • No learned fusion weights                                 │
 │                                                                  │
 │  4. RE-RANKING                                                   │
 │     • No cross-encoder re-ranking                               │
 │     • No LLM-based relevance scoring                            │
 │     • Missing diversity optimization                            │
 │                                                                  │
 │  5. UNIFIED ORCHESTRATION                                        │
 │     • No query router to select retrieval strategy              │
 │     • No cost-aware query planning                              │
 │     • No fallback/cascade mechanisms                            │
 │                                                                  │
 └─────────────────────────────────────────────────────────────────┘
```

---

## 3. The Case for a Unified Semantic Layer

### 3.1 Research Evidence

Recent benchmarks demonstrate the superiority of hybrid approaches:

| Study | Finding |
|-------|---------|
| [Diffbot KG-LM Benchmark](https://www.falkordb.com/blog/graphrag-accuracy-diffbot-falkordb/) | GraphRAG outperforms vector RAG 3.4x for entity-heavy queries |
| [Elastic Hybrid Search](https://www.elastic.co/what-is/hybrid-search) | Hybrid search achieves better NDCG than single-modality |
| [Microsoft Azure Research](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking) | RRF fusion provides 15-30% precision improvements |
| [Neo4j Enterprise](https://neo4j.com/blog/developer/knowledge-graph-vs-vector-rag/) | Knowledge graphs excel at 5+ entity queries where vector fails |

### 3.2 Query Type Analysis

Different query types require different retrieval strategies:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         QUERY TYPE MATRIX                                 │
├────────────────────────┬─────────┬─────────┬──────────┬─────────────────┤
│ Query Type             │ Vector  │ Graph   │ Lexical  │ Best Strategy   │
├────────────────────────┼─────────┼─────────┼──────────┼─────────────────┤
│ "What is machine       │ ✅ Best │ ⚠️ OK   │ ⚠️ OK    │ Vector          │
│  learning?"            │         │         │          │                 │
├────────────────────────┼─────────┼─────────┼──────────┼─────────────────┤
│ "How is Steve Jobs     │ ❌ Weak │ ✅ Best │ ❌ Weak  │ Graph           │
│  related to iPhone?"   │         │         │          │                 │
├────────────────────────┼─────────┼─────────┼──────────┼─────────────────┤
│ "Find documents with   │ ⚠️ OK   │ ❌ N/A  │ ✅ Best  │ Lexical         │
│  error code ERR-2847"  │         │         │          │                 │
├────────────────────────┼─────────┼─────────┼──────────┼─────────────────┤
│ "Summarize the Q3      │ ⚠️ OK   │ ⚠️ OK   │ ⚠️ OK    │ HYBRID          │
│  revenue impact on     │         │         │          │ (All three)     │
│  APAC partnerships"    │         │         │          │                 │
├────────────────────────┼─────────┼─────────┼──────────┼─────────────────┤
│ "Which companies did   │ ❌ Weak │ ✅ Best │ ⚠️ OK    │ Graph + Lexical │
│  Acme Corp acquire     │         │         │          │                 │
│  in 2024?"             │         │         │          │                 │
└────────────────────────┴─────────┴─────────┴──────────┴─────────────────┘
```

### 3.3 The Unified Advantage

By combining all three modalities:

1. **Semantic Coverage**: Vector search finds conceptually related content
2. **Structural Reasoning**: Graph traversal connects entities across documents
3. **Precision Matching**: Lexical search catches exact terms and codes
4. **Robustness**: Fallback when one modality fails

---

## 4. Proposed Architecture

### 4.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PHOENIX AI - SEMANTIC SEARCH LAYER                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         QUERY UNDERSTANDING                                 │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │ │
│  │  │   Intent    │  │   Entity    │  │   Query     │  │    Semantic     │   │ │
│  │  │ Classifier  │  │   Linker    │  │ Decomposer  │  │    Router       │   │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘   │ │
│  └─────────┼────────────────┼────────────────┼──────────────────┼────────────┘ │
│            └────────────────┴────────────────┴──────────────────┘              │
│                                      │                                          │
│                                      ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         RETRIEVAL ORCHESTRATOR                              │ │
│  │                                                                             │ │
│  │   ┌─────────────────────────────────────────────────────────────────────┐  │ │
│  │   │                    PARALLEL RETRIEVAL ENGINES                        │  │ │
│  │   │                                                                      │  │ │
│  │   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │  │ │
│  │   │  │    DENSE     │  │    GRAPH     │  │        SPARSE            │   │  │ │
│  │   │  │   VECTOR     │  │  RETRIEVER   │  │       RETRIEVER          │   │  │ │
│  │   │  │              │  │              │  │                          │   │  │ │
│  │   │  │ • Embedding  │  │ • Entity     │  │ • BM25 (exact match)     │   │  │ │
│  │   │  │   similarity │  │   matching   │  │ • SPLADE (learned        │   │  │ │
│  │   │  │ • FAISS      │  │ • Multi-hop  │  │   sparse)                │   │  │ │
│  │   │  │ • Azure      │  │   traversal  │  │ • TF-IDF fallback        │   │  │ │
│  │   │  │ • Milvus     │  │ • Subgraph   │  │                          │   │  │ │
│  │   │  │              │  │   extraction │  │                          │   │  │ │
│  │   │  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘   │  │ │
│  │   │         │                 │                       │                  │  │ │
│  │   │         └─────────────────┼───────────────────────┘                  │  │ │
│  │   │                           ▼                                          │  │ │
│  │   │              ┌─────────────────────────────┐                         │  │ │
│  │   │              │      SCORE FUSION           │                         │  │ │
│  │   │              │                             │                         │  │ │
│  │   │              │  • Reciprocal Rank Fusion   │                         │  │ │
│  │   │              │  • Weighted Linear Combo    │                         │  │ │
│  │   │              │  • Learned Fusion (ML)      │                         │  │ │
│  │   │              └──────────────┬──────────────┘                         │  │ │
│  │   │                             ▼                                        │  │ │
│  │   │              ┌─────────────────────────────┐                         │  │ │
│  │   │              │       RE-RANKER             │                         │  │ │
│  │   │              │                             │                         │  │ │
│  │   │              │  • Cross-Encoder (BERT)     │                         │  │ │
│  │   │              │  • LLM relevance scoring    │                         │  │ │
│  │   │              │  • Diversity optimization   │                         │  │ │
│  │   │              └──────────────┬──────────────┘                         │  │ │
│  │   └──────────────────────────────┼───────────────────────────────────────┘  │ │
│  │                                  ▼                                          │ │
│  │              ┌─────────────────────────────────────┐                       │ │
│  │              │       CONTEXT ASSEMBLER             │                       │ │
│  │              │                                     │                       │ │
│  │              │  • Chunk deduplication              │                       │ │
│  │              │  • Graph context injection          │                       │ │
│  │              │  • Source attribution               │                       │ │
│  │              │  • Token budget management          │                       │ │
│  │              └──────────────────┬──────────────────┘                       │ │
│  └─────────────────────────────────┼───────────────────────────────────────────┘ │
│                                    ▼                                             │
│                         ┌─────────────────────┐                                  │
│                         │   LLM Generation    │                                  │
│                         │   (GenAIChatClient) │                                  │
│                         └─────────────────────┘                                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Module Breakdown

```
phoenix_ai/
├── semantic_layer/                    # NEW MODULE
│   ├── __init__.py
│   │
│   ├── query_understanding/           # Query preprocessing
│   │   ├── intent_classifier.py       # Simple/complex/multi-hop detection
│   │   ├── entity_linker.py           # Link mentions to KG entities
│   │   ├── query_decomposer.py        # Break complex queries into sub-queries
│   │   └── semantic_router.py         # Route to optimal retrieval strategy
│   │
│   ├── retrievers/                    # Individual retrieval engines
│   │   ├── dense_retriever.py         # Vector search (existing, refactored)
│   │   ├── graph_retriever.py         # Graph traversal (existing, enhanced)
│   │   ├── sparse_retriever.py        # NEW: BM25 + SPLADE
│   │   └── hybrid_retriever.py        # Orchestrates all three
│   │
│   ├── fusion/                        # Score combination
│   │   ├── rrf_fusion.py              # Reciprocal Rank Fusion
│   │   ├── weighted_fusion.py         # Linear combination
│   │   └── learned_fusion.py          # ML-based fusion weights
│   │
│   ├── reranking/                     # Second-stage ranking
│   │   ├── cross_encoder_reranker.py  # BERT-based reranking
│   │   ├── llm_reranker.py            # LLM relevance scoring
│   │   └── diversity_reranker.py      # MMR-style diversity
│   │
│   ├── context/                       # Context assembly
│   │   ├── context_builder.py         # Unified context creation
│   │   ├── graph_context_injector.py  # Add entity relationships
│   │   └── token_budget_manager.py    # Fit within LLM limits
│   │
│   └── unified_inferencer.py          # Main entry point
│
└── existing modules...
```

---

## 5. Component Deep Dive

### 5.1 Sparse Retriever (NEW)

The sparse retriever adds lexical matching capabilities missing from current Phoenix.

```python
class SparseRetriever:
    """
    Sparse retrieval using BM25 and/or SPLADE.

    BM25: Traditional term frequency-based ranking
    SPLADE: Learned sparse representations with term expansion
    """

    def __init__(
        self,
        method: str = "bm25",  # "bm25", "splade", or "hybrid"
        splade_model: str = "naver/splade-cocondenser-ensembledistil",
    ):
        self.method = method
        if method in ["splade", "hybrid"]:
            self.splade = self._load_splade(splade_model)
        if method in ["bm25", "hybrid"]:
            self.bm25_index = None  # Built during indexing

    def index(self, chunks: List[str], chunk_ids: List[str]):
        """Build sparse index from chunks."""
        if self.method in ["bm25", "hybrid"]:
            # BM25 index using rank_bm25 or custom implementation
            tokenized = [self._tokenize(c) for c in chunks]
            self.bm25_index = BM25Okapi(tokenized)
            self.chunk_ids = chunk_ids
            self.chunks = chunks

        if self.method in ["splade", "hybrid"]:
            # SPLADE sparse vectors
            self.splade_vectors = self.splade.encode(chunks)
            self.splade_index = self._build_sparse_index(self.splade_vectors)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search using sparse retrieval."""
        results = []

        if self.method in ["bm25", "hybrid"]:
            bm25_scores = self.bm25_index.get_scores(self._tokenize(query))
            bm25_results = [
                (self.chunk_ids[i], score)
                for i, score in enumerate(bm25_scores)
            ]
            results.extend(bm25_results)

        if self.method in ["splade", "hybrid"]:
            query_vec = self.splade.encode([query])[0]
            splade_results = self.splade_index.search(query_vec, k)
            results.extend(splade_results)

        # Deduplicate and sort
        return self._merge_results(results, k)
```

**Why SPLADE over pure BM25?**

| Aspect | BM25 | SPLADE |
|--------|------|--------|
| Term Matching | Exact only | Learned expansion |
| "car" → "automobile" | ❌ No | ✅ Yes |
| Training Required | No | Yes (pretrained available) |
| Speed | Very fast | Fast |
| Best For | Exact codes, IDs | Semantic + lexical |

### 5.2 Query Understanding Pipeline

```python
class QueryUnderstandingPipeline:
    """
    Analyzes query to determine optimal retrieval strategy.
    """

    def __init__(self, chat_client, knowledge_graph=None):
        self.chat_client = chat_client
        self.knowledge_graph = knowledge_graph
        self.intent_classifier = IntentClassifier(chat_client)
        self.entity_linker = EntityLinker(knowledge_graph)
        self.query_decomposer = QueryDecomposer(chat_client)

    def analyze(self, query: str) -> QueryAnalysis:
        """Full query analysis pipeline."""

        # 1. Classify intent
        intent = self.intent_classifier.classify(query)
        # Returns: "factual", "relational", "analytical", "navigational"

        # 2. Extract and link entities
        entities = self.entity_linker.extract_and_link(query)
        # Returns: [(mention, entity_id, confidence), ...]

        # 3. Determine complexity
        complexity = self._assess_complexity(query, entities)
        # Returns: "simple", "moderate", "complex", "multi-hop"

        # 4. Decompose if complex
        sub_queries = []
        if complexity in ["complex", "multi-hop"]:
            sub_queries = self.query_decomposer.decompose(query)

        # 5. Determine retrieval strategy
        strategy = self._select_strategy(intent, complexity, entities)

        return QueryAnalysis(
            original_query=query,
            intent=intent,
            complexity=complexity,
            linked_entities=entities,
            sub_queries=sub_queries,
            retrieval_strategy=strategy,
        )

    def _select_strategy(self, intent, complexity, entities) -> RetrievalStrategy:
        """
        Select optimal retrieval configuration.
        """
        strategy = RetrievalStrategy()

        # Relational queries need graph
        if intent == "relational" or len(entities) >= 2:
            strategy.use_graph = True
            strategy.graph_weight = 0.5

        # Complex queries need all modalities
        if complexity in ["complex", "multi-hop"]:
            strategy.use_dense = True
            strategy.use_sparse = True
            strategy.use_graph = True
            strategy.fusion_method = "rrf"

        # Simple factual queries can use primarily vector
        elif complexity == "simple" and intent == "factual":
            strategy.use_dense = True
            strategy.use_sparse = True  # For exact matches
            strategy.use_graph = False
            strategy.fusion_method = "weighted"

        return strategy
```

### 5.3 Semantic Router

```python
class SemanticRouter:
    """
    Routes queries to appropriate retrieval paths based on analysis.

    Routing Logic:
    ─────────────────────────────────────────────────────────────────

    Query: "What is machine learning?"
      → SIMPLE FACTUAL → Dense + Sparse, no graph
      → Fast path, low cost

    Query: "How did Apple's acquisition of NeXT lead to the iPhone?"
      → MULTI-HOP RELATIONAL → All modalities + decomposition
      → Slow path, high accuracy

    Query: "Find error ERR-7829 in production logs"
      → NAVIGATIONAL → Sparse primary, dense secondary
      → Exact match priority

    Query: "Summarize Q3 revenue trends across regions"
      → ANALYTICAL → Dense + Graph (for entity relationships)
      → Aggregation support
    ─────────────────────────────────────────────────────────────────
    """

    ROUTING_TABLE = {
        ("factual", "simple"): {
            "retrievers": ["dense", "sparse"],
            "weights": {"dense": 0.7, "sparse": 0.3},
            "fusion": "weighted",
            "rerank": False,
        },
        ("factual", "moderate"): {
            "retrievers": ["dense", "sparse", "graph"],
            "weights": {"dense": 0.5, "sparse": 0.2, "graph": 0.3},
            "fusion": "rrf",
            "rerank": True,
        },
        ("relational", "simple"): {
            "retrievers": ["graph", "dense"],
            "weights": {"graph": 0.6, "dense": 0.4},
            "fusion": "weighted",
            "rerank": False,
        },
        ("relational", "complex"): {
            "retrievers": ["graph", "dense", "sparse"],
            "weights": {"graph": 0.5, "dense": 0.3, "sparse": 0.2},
            "fusion": "rrf",
            "rerank": True,
            "decompose": True,
        },
        ("navigational", "simple"): {
            "retrievers": ["sparse", "dense"],
            "weights": {"sparse": 0.8, "dense": 0.2},
            "fusion": "weighted",
            "rerank": False,
        },
        # ... more routing rules
    }

    def route(self, analysis: QueryAnalysis) -> RetrievalPlan:
        """Create execution plan based on query analysis."""
        key = (analysis.intent, analysis.complexity)
        config = self.ROUTING_TABLE.get(key, self.DEFAULT_CONFIG)

        return RetrievalPlan(
            retrievers=config["retrievers"],
            weights=config["weights"],
            fusion_method=config["fusion"],
            use_reranker=config.get("rerank", False),
            decompose=config.get("decompose", False),
            sub_queries=analysis.sub_queries if config.get("decompose") else [],
        )
```

---

## 6. Fusion Algorithms

### 6.1 Reciprocal Rank Fusion (RRF)

RRF is the recommended fusion method for combining results from different retrieval systems.

```python
class RRFFusion:
    """
    Reciprocal Rank Fusion implementation.

    Formula: RRF_score(d) = Σ 1 / (k + rank_i(d))

    Where:
    - d is a document
    - k is a constant (typically 60)
    - rank_i(d) is the rank of document d in result list i

    Advantages:
    - No score normalization needed
    - Robust across different scoring scales
    - Proven effectiveness in hybrid search
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        result_lists: Dict[str, List[Tuple[str, float]]]
    ) -> List[Tuple[str, float]]:
        """
        Fuse multiple ranked result lists.

        Args:
            result_lists: Dict mapping retriever name to [(doc_id, score), ...]

        Returns:
            Fused ranked list: [(doc_id, rrf_score), ...]
        """
        doc_scores = defaultdict(float)

        for retriever_name, results in result_lists.items():
            for rank, (doc_id, _) in enumerate(results):
                # RRF formula: 1 / (k + rank)
                rrf_contribution = 1.0 / (self.k + rank + 1)  # +1 for 0-indexed
                doc_scores[doc_id] += rrf_contribution

        # Sort by fused score
        fused = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return fused

    def fuse_with_weights(
        self,
        result_lists: Dict[str, List[Tuple[str, float]]],
        weights: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """
        Weighted RRF - applies per-retriever weights.

        Useful when some retrievers are more reliable for certain query types.
        """
        doc_scores = defaultdict(float)

        for retriever_name, results in result_lists.items():
            weight = weights.get(retriever_name, 1.0)
            for rank, (doc_id, _) in enumerate(results):
                rrf_contribution = weight / (self.k + rank + 1)
                doc_scores[doc_id] += rrf_contribution

        fused = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return fused
```

### 6.2 Fusion Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FUSION METHOD COMPARISON                             │
├──────────────────┬────────────────┬────────────────┬────────────────────────┤
│ Method           │ Pros           │ Cons           │ Best For               │
├──────────────────┼────────────────┼────────────────┼────────────────────────┤
│ Simple Average   │ Easy           │ Score scale    │ Prototyping            │
│                  │                │ mismatch       │                        │
├──────────────────┼────────────────┼────────────────┼────────────────────────┤
│ Weighted Average │ Tunable        │ Requires       │ Known retriever        │
│                  │                │ calibration    │ reliability            │
├──────────────────┼────────────────┼────────────────┼────────────────────────┤
│ RRF              │ No tuning,     │ Ignores score  │ General purpose        │
│                  │ robust         │ magnitudes     │ (RECOMMENDED)          │
├──────────────────┼────────────────┼────────────────┼────────────────────────┤
│ Learned Fusion   │ Optimal for    │ Needs training │ Production with        │
│ (ML-based)       │ your data      │ data           │ labeled data           │
├──────────────────┼────────────────┼────────────────┼────────────────────────┤
│ Cascade          │ Cost efficient │ May miss good  │ High-volume,           │
│                  │                │ results        │ cost-sensitive         │
└──────────────────┴────────────────┴────────────────┴────────────────────────┘
```

---

## 7. Query Understanding & Routing

### 7.1 Intent Classification

```python
class IntentClassifier:
    """
    Classifies query intent to guide retrieval strategy.

    Intent Types:
    - FACTUAL: Direct information lookup ("What is X?")
    - RELATIONAL: Connections between entities ("How is X related to Y?")
    - ANALYTICAL: Aggregation/comparison ("Compare X and Y trends")
    - NAVIGATIONAL: Find specific item ("Show me document ABC-123")
    - PROCEDURAL: How-to instructions ("How do I configure X?")
    """

    INTENT_PROMPT = """
    Classify the following query into one of these categories:
    - FACTUAL: Questions seeking specific facts or definitions
    - RELATIONAL: Questions about relationships between entities
    - ANALYTICAL: Questions requiring analysis, comparison, or aggregation
    - NAVIGATIONAL: Requests to find a specific document or resource
    - PROCEDURAL: How-to questions seeking step-by-step guidance

    Query: {query}

    Respond with only the category name.
    """

    def classify(self, query: str) -> str:
        response = self.chat_client.chat(
            user_input=self.INTENT_PROMPT.format(query=query),
            max_tokens=20,
        )
        return response.strip().upper()
```

### 7.2 Query Decomposition for Multi-hop

```python
class QueryDecomposer:
    """
    Breaks complex multi-hop questions into simpler sub-questions.

    Example:
    ─────────────────────────────────────────────────────────────────
    Input: "How did Apple's acquisition of NeXT influence the iPhone's design?"

    Output:
    1. "When did Apple acquire NeXT?"
    2. "Who founded NeXT and what was their role at Apple?"
    3. "What technology from NeXT was used in Apple products?"
    4. "What design elements of iPhone came from NeXT technology?"
    ─────────────────────────────────────────────────────────────────

    Each sub-question is answered independently, and results are
    synthesized to answer the original question.
    """

    DECOMPOSITION_PROMPT = """
    Break down this complex question into simpler sub-questions that can be
    answered independently. Each sub-question should be self-contained.

    Original Question: {query}

    Return a JSON list of sub-questions:
    ["sub-question 1", "sub-question 2", ...]

    Keep it to 2-5 sub-questions maximum.
    """

    def decompose(self, query: str) -> List[str]:
        response = self.chat_client.chat(
            user_input=self.DECOMPOSITION_PROMPT.format(query=query),
            max_tokens=500,
        )
        return self._parse_json_list(response)
```

### 7.3 Entity Linking

```python
class EntityLinker:
    """
    Links entity mentions in query to knowledge graph nodes.

    Process:
    1. NER: Detect entity mentions in query
    2. Candidate Generation: Find possible KG matches
    3. Disambiguation: Select best match using context

    Example:
    ─────────────────────────────────────────────────────────────────
    Query: "What products did Apple release after Jobs returned?"

    Detected Mentions:
    - "Apple" → [Apple Inc. (ORG), Apple fruit (FOOD)]
    - "Jobs" → [Steve Jobs (PERSON), job listings (CONCEPT)]

    Disambiguation (using context):
    - "Apple" → Apple Inc. (confidence: 0.95)
    - "Jobs" → Steve Jobs (confidence: 0.92)

    Output: Entity IDs for knowledge graph traversal
    ─────────────────────────────────────────────────────────────────
    """

    def extract_and_link(self, query: str) -> List[LinkedEntity]:
        # Step 1: Extract mentions using LLM
        mentions = self._extract_mentions(query)

        linked = []
        for mention in mentions:
            # Step 2: Find candidates in knowledge graph
            candidates = self.knowledge_graph.find_entities_by_name(
                mention.text, fuzzy=True
            )

            if not candidates:
                continue

            # Step 3: Disambiguate using context
            best_match = self._disambiguate(mention, candidates, query)
            if best_match:
                linked.append(LinkedEntity(
                    mention=mention.text,
                    entity_id=best_match.id,
                    entity_name=best_match.name,
                    entity_type=best_match.entity_type,
                    confidence=best_match.score,
                ))

        return linked
```

---

## 8. Implementation Roadmap

### Phase 1: Foundation (2-3 weeks)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: SPARSE RETRIEVAL + RRF FUSION                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Week 1:                                                                 │
│  ☐ Implement BM25 index using rank-bm25 library                         │
│  ☐ Add BM25 to VectorEmbedding pipeline (optional index type)          │
│  ☐ Create SparseRetriever class                                        │
│                                                                          │
│  Week 2:                                                                 │
│  ☐ Implement RRF fusion algorithm                                       │
│  ☐ Create BasicHybridRetriever (dense + sparse)                        │
│  ☐ Add hybrid mode to RAGInferencer                                    │
│                                                                          │
│  Week 3:                                                                 │
│  ☐ Add SPLADE support (optional, for learned sparse)                   │
│  ☐ Benchmark hybrid vs vector-only                                     │
│  ☐ Documentation and examples                                          │
│                                                                          │
│  Deliverables:                                                           │
│  • BM25/SPLADE sparse retriever                                         │
│  • RRF fusion module                                                    │
│  • BasicHybridInferencer with dense + sparse                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Query Understanding (2-3 weeks)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: QUERY ANALYSIS + ROUTING                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Week 4:                                                                 │
│  ☐ Implement IntentClassifier                                           │
│  ☐ Implement query complexity scoring                                   │
│  ☐ Create SemanticRouter with routing table                            │
│                                                                          │
│  Week 5:                                                                 │
│  ☐ Implement EntityLinker (integrate with KnowledgeGraph)              │
│  ☐ Add entity disambiguation using context                             │
│  ☐ Link query entities to graph nodes                                  │
│                                                                          │
│  Week 6:                                                                 │
│  ☐ Implement QueryDecomposer for multi-hop                             │
│  ☐ Create QueryUnderstandingPipeline orchestrator                      │
│  ☐ Integration tests                                                    │
│                                                                          │
│  Deliverables:                                                           │
│  • Query understanding pipeline                                         │
│  • Smart routing based on query type                                    │
│  • Entity linking to knowledge graph                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 3: Full Integration (2-3 weeks)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: UNIFIED SEMANTIC LAYER                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Week 7:                                                                 │
│  ☐ Create UnifiedSemanticRetriever                                      │
│  ☐ Integrate all three retrievers (dense, sparse, graph)               │
│  ☐ Dynamic strategy selection based on routing                         │
│                                                                          │
│  Week 8:                                                                 │
│  ☐ Add cross-encoder re-ranking (optional)                             │
│  ☐ Implement enhanced context builder                                  │
│  ☐ Create SemanticLayerInferencer                                      │
│                                                                          │
│  Week 9:                                                                 │
│  ☐ Performance optimization                                             │
│  ☐ Comprehensive benchmarking                                          │
│  ☐ Documentation and migration guide                                   │
│                                                                          │
│  Deliverables:                                                           │
│  • SemanticLayerInferencer (fully unified)                              │
│  • Complete query → retrieval → generation pipeline                    │
│  • Benchmark results vs existing inferencers                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 4: Advanced Features (Optional, 2-4 weeks)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: ADVANCED CAPABILITIES                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ☐ Learned fusion weights (train on user feedback)                      │
│  ☐ Query result caching with semantic keys                              │
│  ☐ Streaming retrieval results                                          │
│  ☐ Multi-hop question decomposition with parallel sub-queries          │
│  ☐ LLM-based re-ranking for top results                                │
│  ☐ Feedback loop for continuous improvement                            │
│  ☐ A/B testing framework for retrieval strategies                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Comparison: Before vs After

### 9.1 Feature Matrix

| Capability | Current RAGInferencer | Current GraphRAGInferencer | **SemanticLayerInferencer** |
|------------|:---------------------:|:--------------------------:|:---------------------------:|
| Dense Vector Search | ✅ | ✅ | ✅ |
| Sparse/BM25 Search | ❌ | ❌ | ✅ |
| SPLADE Learned Sparse | ❌ | ❌ | ✅ |
| Knowledge Graph | ❌ | ✅ | ✅ |
| RRF Fusion | ❌ | ❌ | ✅ |
| Query Intent Classification | ❌ | ❌ | ✅ |
| Entity Linking | ❌ | Basic | ✅ Full |
| Query Decomposition | ❌ | ❌ | ✅ |
| Semantic Routing | ❌ | ❌ | ✅ |
| Cross-Encoder Reranking | ❌ | ❌ | ✅ (optional) |
| Adaptive Strategy | ❌ | ❌ | ✅ |

### 9.2 Query Handling Comparison

```
QUERY: "Which companies that Softbank invested in also have partnerships
        with Microsoft, and what products did they collaborate on?"

┌─────────────────────────────────────────────────────────────────────────┐
│                         CURRENT: RAGInferencer                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Embed query as single vector                                        │
│  2. Find top-k similar chunks                                           │
│  3. May retrieve chunks about:                                          │
│     - Softbank investments (partial)                                    │
│     - Microsoft partnerships (partial)                                  │
│     - Unrelated but semantically similar text                          │
│                                                                          │
│  Problem: No connection between Softbank → Company → Microsoft          │
│  Result: Incomplete or hallucinated answer                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      CURRENT: GraphRAGInferencer                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Extract entities: Softbank, Microsoft                               │
│  2. Vector search + graph expansion                                     │
│  3. Find: Softbank → invested_in → [Companies]                         │
│           Microsoft → partners_with → [Companies]                       │
│  4. Intersection: Companies in both sets                                │
│                                                                          │
│  Better: Captures relationships                                         │
│  Gap: May miss exact product names not in graph                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      NEW: SemanticLayerInferencer                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. QUERY UNDERSTANDING                                                  │
│     - Intent: RELATIONAL                                                │
│     - Complexity: MULTI-HOP                                             │
│     - Entities: [Softbank, Microsoft]                                   │
│     - Decompose into:                                                   │
│       Q1: "Which companies did Softbank invest in?"                     │
│       Q2: "Which companies partner with Microsoft?"                     │
│       Q3: "What products resulted from these partnerships?"             │
│                                                                          │
│  2. ROUTING                                                              │
│     - Primary: Graph (for relationships)                                │
│     - Secondary: Sparse (for exact product names)                       │
│     - Tertiary: Dense (for semantic context)                            │
│                                                                          │
│  3. PARALLEL RETRIEVAL                                                   │
│     Graph: Softbank →invested→ [ARM, WeWork, Uber, ...]                │
│            Microsoft →partners→ [OpenAI, SAP, ...]                      │
│            Intersection: [OpenAI, ...]                                  │
│                                                                          │
│     Sparse: Exact match for product names                               │
│             "GPT-4", "Azure OpenAI Service", ...                        │
│                                                                          │
│     Dense: Semantic chunks about collaborations                         │
│                                                                          │
│  4. RRF FUSION                                                           │
│     Merge all results, weight graph higher for this query              │
│                                                                          │
│  5. CONTEXT ASSEMBLY                                                     │
│     Documents + Graph relationships + Entity context                   │
│                                                                          │
│  Result: Complete, accurate, multi-sourced answer                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Performance Benchmarks

### 10.1 Expected Improvements

Based on research benchmarks:

| Metric | Vector Only | Hybrid (Vec+Sparse) | Full Semantic Layer |
|--------|-------------|---------------------|---------------------|
| Precision@10 | Baseline | +15-30% | +25-40% |
| Recall@100 | Baseline | +10-20% | +20-35% |
| Multi-hop Accuracy | 30-40% | 35-45% | 70-85% |
| Entity-heavy Queries | 40-50% | 50-60% | 85-95% |
| Exact Match Queries | 50-60% | 90-95% | 90-95% |

### 10.2 Latency Considerations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LATENCY BREAKDOWN                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SIMPLE QUERY (routed to dense + sparse only):                          │
│  ├── Query embedding:        ~50ms                                      │
│  ├── Dense search:           ~30ms                                      │
│  ├── Sparse search:          ~20ms (parallel)                           │
│  ├── RRF fusion:             ~5ms                                       │
│  ├── Context build:          ~10ms                                      │
│  └── TOTAL:                  ~115ms (retrieval only)                    │
│                                                                          │
│  COMPLEX QUERY (all modalities + reranking):                            │
│  ├── Query understanding:    ~200ms (LLM call)                          │
│  ├── Query decomposition:    ~300ms (if needed)                         │
│  ├── Dense search:           ~30ms                                      │
│  ├── Sparse search:          ~20ms (parallel)                           │
│  ├── Graph traversal:        ~100ms (parallel)                          │
│  ├── RRF fusion:             ~10ms                                      │
│  ├── Reranking:              ~150ms (cross-encoder)                     │
│  ├── Context build:          ~20ms                                      │
│  └── TOTAL:                  ~830ms (retrieval only)                    │
│                                                                          │
│  OPTIMIZATION STRATEGIES:                                                │
│  • Cache query understanding for similar queries                        │
│  • Skip reranking for high-confidence results                          │
│  • Parallel execution of all retrievers                                │
│  • Early termination if top results are confident                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. References

### Research Papers & Articles

1. [Research on KG-RAG Model](https://www.nature.com/articles/s41598-025-21222-z) - Nature Scientific Reports on knowledge graph enhanced RAG
2. [Graph RAG Survey](https://openreview.net/pdf?id=9FJiOMuZkr) - Comprehensive survey of graph retrieval augmented generation
3. [HopRAG: Multi-Hop Reasoning](https://arxiv.org/html/2502.12442v1) - Logic-aware RAG with graph-structured knowledge exploration
4. [SPLADE v2](https://arxiv.org/abs/2109.10086) - Sparse lexical and expansion model for information retrieval

### Industry Resources

5. [Neo4j Essential GraphRAG](https://go.neo4j.com/rs/710-RRC-335/images/Essential-GraphRAG.pdf) - Enterprise GraphRAG implementation guide
6. [Weaviate: Graph RAG Guide](https://weaviate.io/blog/graph-rag) - When and how to use GraphRAG
7. [Elastic Hybrid Search](https://www.elastic.co/what-is/hybrid-search) - Comprehensive hybrid search guide
8. [Azure AI Search RRF](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking) - Microsoft's RRF implementation

### Benchmarks & Comparisons

9. [GraphRAG vs Vector RAG Comparison](https://www.meilisearch.com/blog/graph-rag-vs-vector-rag) - Side-by-side analysis
10. [FalkorDB GraphRAG Accuracy Benchmark](https://www.falkordb.com/blog/graphrag-accuracy-diffbot-falkordb/) - Diffbot KG-LM benchmark results
11. [Neo4j Knowledge Graph vs Vector RAG](https://neo4j.com/blog/developer/knowledge-graph-vs-vector-rag/) - Optimization levers and benchmarks

### Implementation Guides

12. [Pinecone SPLADE Guide](https://www.pinecone.io/learn/splade/) - SPLADE for sparse vector search
13. [OpenSearch RRF Introduction](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/) - RRF algorithm explanation
14. [Superlinked: KG for RAG Performance](https://superlinked.com/vectorhub/articles/improving-rag-performance-knowledge-graphs) - Practical knowledge graph integration

---

## Appendix A: Proposed Class Diagram

```
                              ┌──────────────────────────────┐
                              │   SemanticLayerInferencer    │
                              │                              │
                              │ + infer(query, ...)          │
                              │ + configure(settings)        │
                              └──────────────┬───────────────┘
                                             │
                         ┌───────────────────┼───────────────────┐
                         │                   │                   │
                         ▼                   ▼                   ▼
          ┌──────────────────────┐ ┌─────────────────┐ ┌────────────────────┐
          │QueryUnderstandingPipe│ │RetrievalOrchest │ │  ContextAssembler  │
          │                      │ │                 │ │                    │
          │ + analyze(query)     │ │ + retrieve()    │ │ + build_context()  │
          │ + route(analysis)    │ │ + fuse()        │ │ + inject_graph()   │
          └──────────┬───────────┘ │ + rerank()      │ │ + manage_tokens()  │
                     │             └────────┬────────┘ └────────────────────┘
        ┌────────────┼────────────┐         │
        │            │            │         │
        ▼            ▼            ▼         │
┌────────────┐┌────────────┐┌─────────────┐ │
│  Intent    ││  Entity    ││   Query     │ │
│ Classifier ││  Linker    ││ Decomposer  │ │
└────────────┘└────────────┘└─────────────┘ │
                                            │
              ┌─────────────────────────────┤
              │             │               │
              ▼             ▼               ▼
      ┌─────────────┐┌─────────────┐┌─────────────┐
      │   Dense     ││   Sparse    ││   Graph     │
      │  Retriever  ││  Retriever  ││  Retriever  │
      │             ││             ││             │
      │ (existing)  ││ (NEW)       ││ (existing)  │
      └─────────────┘└─────────────┘└─────────────┘
              │             │               │
              └─────────────┼───────────────┘
                            ▼
                    ┌─────────────┐
                    │  RRFFusion  │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  Reranker   │
                    │ (optional)  │
                    └─────────────┘
```

---

## Appendix B: Configuration Example

```python
from phoenix_ai.semantic_layer import (
    SemanticLayerInferencer,
    SemanticLayerConfig,
)

# Full configuration
config = SemanticLayerConfig(
    # Retriever settings
    dense_retriever={
        "enabled": True,
        "index_type": "faiss",  # or "azure", "milvus"
        "top_k": 20,
    },
    sparse_retriever={
        "enabled": True,
        "method": "bm25",  # or "splade", "hybrid"
        "top_k": 20,
    },
    graph_retriever={
        "enabled": True,
        "max_depth": 2,
        "top_k": 10,
    },

    # Fusion settings
    fusion={
        "method": "rrf",  # or "weighted", "learned"
        "rrf_k": 60,
        "weights": {  # Used if method="weighted"
            "dense": 0.5,
            "sparse": 0.2,
            "graph": 0.3,
        },
    },

    # Query understanding
    query_understanding={
        "intent_classification": True,
        "entity_linking": True,
        "query_decomposition": True,
        "semantic_routing": True,
    },

    # Reranking
    reranking={
        "enabled": True,
        "method": "cross_encoder",  # or "llm", "none"
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_k": 10,  # Rerank top 10 after fusion
    },

    # Performance
    performance={
        "parallel_retrieval": True,
        "cache_query_analysis": True,
        "early_termination": True,
    },
)

# Create inferencer
inferencer = SemanticLayerInferencer(
    embedding_client=embedding_client,
    chat_client=chat_client,
    knowledge_graph=kg,  # Optional
    config=config,
)

# Use it
result = inferencer.infer(
    question="How is Apple's M1 chip related to their ARM acquisition?",
    index_path="output/docs.index",
)
```

---

*Document Version: 1.0*
*Last Updated: February 2026*
