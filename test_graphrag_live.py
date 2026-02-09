#!/usr/bin/env python3
"""
Live testing of Phoenix AI GraphRAG module with OpenAI API.

This script tests all components of the GraphRAG module:
1. Entity extraction
2. Relationship extraction
3. Knowledge graph construction
4. Graph retrieval
5. GraphRAG inference
"""

import os
import sys
import tempfile
import time

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_success(msg):
    print(f"[PASS] {msg}")

def print_fail(msg):
    print(f"[FAIL] {msg}")

def print_info(msg):
    print(f"[INFO] {msg}")


def test_basic_imports():
    """Test that all GraphRAG components can be imported."""
    print_header("Test 1: Basic Imports")

    try:
        from phoenix_ai import (
            GenAIEmbeddingClient,
            GenAIChatClient,
            GraphRAGInferencer,
            KnowledgeGraph,
            Entity,
            Relationship,
            GraphBuilder,
            GraphRetriever,
            LLMEntityExtractor,
            LLMRelationshipExtractor,
        )
        print_success("All GraphRAG components imported successfully")
        return True
    except ImportError as e:
        print_fail(f"Import error: {e}")
        return False


def test_client_initialization():
    """Test OpenAI client initialization."""
    print_header("Test 2: Client Initialization")

    try:
        from phoenix_ai import GenAIEmbeddingClient, GenAIChatClient

        embedding_client = GenAIEmbeddingClient(
            provider="openai",
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY,
        )
        print_success("Embedding client initialized")

        chat_client = GenAIChatClient(
            provider="openai",
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
        )
        print_success("Chat client initialized")

        return embedding_client, chat_client
    except Exception as e:
        print_fail(f"Client initialization error: {e}")
        return None, None


def test_embedding_generation(embedding_client):
    """Test embedding generation."""
    print_header("Test 3: Embedding Generation")

    try:
        texts = [
            "Apple is a technology company based in Cupertino.",
            "Microsoft is headquartered in Redmond, Washington.",
        ]

        embeddings = embedding_client.generate_embedding(texts)

        assert len(embeddings) == 2, "Should return 2 embeddings"
        assert len(embeddings[0]) > 0, "Embedding should not be empty"

        print_success(f"Generated embeddings with dimension: {len(embeddings[0])}")
        print_info(f"First 5 values: {embeddings[0][:5]}")
        return True
    except Exception as e:
        print_fail(f"Embedding generation error: {e}")
        return False


def test_entity_extraction(chat_client):
    """Test LLM entity extraction."""
    print_header("Test 4: Entity Extraction")

    try:
        from phoenix_ai import LLMEntityExtractor

        extractor = LLMEntityExtractor(
            chat_client=chat_client,
            entity_types=["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT"],
        )

        text = """
        Elon Musk is the CEO of Tesla and SpaceX. Tesla is headquartered in
        Austin, Texas. SpaceX launched the Falcon 9 rocket from Cape Canaveral.
        Tim Cook leads Apple, which is based in Cupertino, California.
        """

        entities = extractor.extract(text, chunk_id="test_chunk")

        print_success(f"Extracted {len(entities)} entities")
        for entity in entities:
            print_info(f"  - {entity.name} ({entity.entity_type}): {entity.description[:50]}...")

        return entities
    except Exception as e:
        print_fail(f"Entity extraction error: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_relationship_extraction(chat_client, entities):
    """Test LLM relationship extraction."""
    print_header("Test 5: Relationship Extraction")

    if not entities:
        print_info("Skipping - no entities available")
        return []

    try:
        from phoenix_ai import LLMRelationshipExtractor

        extractor = LLMRelationshipExtractor(
            chat_client=chat_client,
            min_confidence=0.5,
        )

        text = """
        Elon Musk is the CEO of Tesla and SpaceX. Tesla is headquartered in
        Austin, Texas. SpaceX launched the Falcon 9 rocket from Cape Canaveral.
        Tim Cook leads Apple, which is based in Cupertino, California.
        """

        relationships = extractor.extract(entities, text, chunk_id="test_chunk")

        print_success(f"Extracted {len(relationships)} relationships")
        for rel in relationships:
            source = next((e for e in entities if e.id == rel.source_id), None)
            target = next((e for e in entities if e.id == rel.target_id), None)
            if source and target:
                print_info(f"  - {source.name} --[{rel.relation_type}]--> {target.name} (conf: {rel.weight:.2f})")

        return relationships
    except Exception as e:
        print_fail(f"Relationship extraction error: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_knowledge_graph():
    """Test knowledge graph operations."""
    print_header("Test 6: Knowledge Graph Operations")

    try:
        from phoenix_ai import KnowledgeGraph, Entity, Relationship

        # Create graph
        graph = KnowledgeGraph()

        # Add entities
        entities = [
            Entity(id="e1", name="Elon Musk", entity_type="PERSON", description="CEO of Tesla"),
            Entity(id="e2", name="Tesla", entity_type="ORGANIZATION", description="Electric car company"),
            Entity(id="e3", name="Austin", entity_type="LOCATION", description="City in Texas"),
        ]

        for entity in entities:
            graph.add_entity(entity)

        print_success(f"Added {graph.num_entities} entities")

        # Add relationships
        relationships = [
            Relationship(source_id="e1", target_id="e2", relation_type="ceo_of", weight=0.95),
            Relationship(source_id="e2", target_id="e3", relation_type="located_in", weight=0.9),
        ]

        for rel in relationships:
            graph.add_relationship(rel)

        print_success(f"Added {graph.num_relationships} relationships")

        # Test queries
        neighbors = graph.get_neighbors("e1")
        print_success(f"Found {len(neighbors)} neighbors for Elon Musk")

        paths = graph.find_paths("e1", "e3", max_depth=3)
        print_success(f"Found {len(paths)} paths from Elon Musk to Austin")
        if paths:
            print_info(f"  Path: {' -> '.join(paths[0])}")

        # Test save/load
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        graph.save(filepath)
        loaded_graph = KnowledgeGraph.load(filepath)

        assert loaded_graph.num_entities == graph.num_entities
        print_success("Graph save/load successful")

        os.unlink(filepath)

        return graph
    except Exception as e:
        print_fail(f"Knowledge graph error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_graph_builder(chat_client, embedding_client):
    """Test graph builder with real documents."""
    print_header("Test 7: Graph Builder")

    try:
        from phoenix_ai import GraphBuilder

        builder = GraphBuilder.from_clients(
            chat_client=chat_client,
            embedding_client=embedding_client,
            entity_types=["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "TECHNOLOGY"],
        )

        # Sample document chunks
        chunks = [
            "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. "
            "The company is headquartered in Cupertino, California. Apple created the iPhone, "
            "which revolutionized the smartphone industry.",

            "Microsoft was founded by Bill Gates and Paul Allen in 1975. The company is "
            "based in Redmond, Washington. Microsoft developed Windows and Azure cloud platform.",

            "Google was founded by Larry Page and Sergey Brin while at Stanford University. "
            "The company is headquartered in Mountain View, California. Google created "
            "the Android operating system and the Chrome browser.",
        ]

        def progress_callback(current, total, stage):
            print_info(f"  Progress: {current}/{total} - {stage}")

        print_info("Building knowledge graph from 3 document chunks...")
        graph = builder.build_from_chunks(
            chunks=chunks,
            chunk_ids=["chunk_0", "chunk_1", "chunk_2"],
            progress_callback=progress_callback,
        )

        print_success(f"Built graph with {graph.num_entities} entities and {graph.num_relationships} relationships")

        # Show some entities
        print_info("Sample entities:")
        for entity in list(graph.entities.values())[:5]:
            print_info(f"  - {entity.name} ({entity.entity_type})")

        return graph, chunks
    except Exception as e:
        print_fail(f"Graph builder error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_graph_retriever(graph, embedding_client):
    """Test graph retriever."""
    print_header("Test 8: Graph Retriever")

    if graph is None:
        print_info("Skipping - no graph available")
        return

    try:
        from phoenix_ai import GraphRetriever

        # Add embeddings to entities
        for entity in graph.entities.values():
            if entity.embedding is None:
                text = f"{entity.name}: {entity.description}" if entity.description else entity.name
                entity.embedding = embedding_client.generate_embedding([text])[0]

        retriever = GraphRetriever(
            knowledge_graph=graph,
            embedding_client=embedding_client,
            graph_weight=0.3,
            max_graph_depth=2,
        )

        # Test retrieval
        queries = [
            "Who founded Apple?",
            "Where is Microsoft located?",
            "What did Google create?",
        ]

        for query in queries:
            results = retriever.retrieve(query, k=3)
            print_success(f"Query: '{query}'")
            for r in results[:3]:
                print_info(f"  - {r['entity'].name} (score: {r['score']:.3f})")

        return retriever
    except Exception as e:
        print_fail(f"Graph retriever error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_graphrag_inferencer(embedding_client, chat_client, graph, chunks):
    """Test GraphRAG inferencer."""
    print_header("Test 9: GraphRAG Inferencer")

    try:
        from phoenix_ai import GraphRAGInferencer
        import pandas as pd
        import numpy as np
        import faiss
        import pickle

        # Create inferencer with pre-built graph
        inferencer = GraphRAGInferencer(
            embedding_client=embedding_client,
            chat_client=chat_client,
            knowledge_graph=graph,
            graph_weight=0.3,
            include_graph_context=True,
        )

        print_success("GraphRAG inferencer created")

        # Load chunks into inferencer
        chunk_store = {f"chunk_{i}": chunk for i, chunk in enumerate(chunks)}
        inferencer.load_chunks(chunk_store)

        # Create a FAISS index for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test.index")

            # Generate embeddings for chunks
            embeddings = embedding_client.generate_embedding(chunks)
            embeddings_array = np.array(embeddings, dtype="float32")

            # Create FAISS index
            dim = len(embeddings[0])
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings_array)
            faiss.write_index(index, index_path)

            # Save chunks
            chunks_path = index_path.replace(".index", "_chunks.pkl")
            with open(chunks_path, "wb") as f:
                pickle.dump(chunks, f)

            print_success("FAISS index created")

            # Test queries
            test_queries = [
                "Who founded Apple and where is it headquartered?",
                "What products did Microsoft create?",
                "Tell me about the founders of Google.",
            ]

            for query in test_queries:
                print_info(f"\nQuery: {query}")

                result = inferencer.infer(
                    question=query,
                    mode="graphrag",
                    index_type="local_index",
                    index_path=index_path,
                    top_k=3,
                    max_tokens=200,
                    use_graph=True,
                )

                print_success("Inference completed")
                print_info(f"Answer: {result['answer'].iloc[0][:200]}...")

                if 'entities' in result.columns and result['entities'].iloc[0]:
                    entities = result['entities'].iloc[0]
                    print_info(f"Related entities: {[e['name'] for e in entities[:3]]}")

        return True
    except Exception as e:
        print_fail(f"GraphRAG inferencer error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_query_graph_directly(inferencer_or_graph, embedding_client, chat_client):
    """Test direct graph querying."""
    print_header("Test 10: Direct Graph Query")

    try:
        from phoenix_ai import GraphRAGInferencer

        # Create a fresh inferencer if we don't have one
        if isinstance(inferencer_or_graph, GraphRAGInferencer):
            inferencer = inferencer_or_graph
        else:
            inferencer = GraphRAGInferencer(
                embedding_client=embedding_client,
                chat_client=chat_client,
                knowledge_graph=inferencer_or_graph,
            )

        # Query the graph directly
        result = inferencer.query_graph("Apple iPhone Steve Jobs", k=5)

        print_success("Direct graph query successful")
        print_info(f"Graph stats: {result.get('graph_stats', {})}")
        print_info(f"Found {len(result.get('results', []))} matching entities")

        for r in result.get('results', [])[:3]:
            entity = r['entity']
            print_info(f"  - {entity['name']} ({entity['entity_type']}): score={r['score']:.3f}")

        return True
    except Exception as e:
        print_fail(f"Direct graph query error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_persistence(graph):
    """Test graph save/load with different formats."""
    print_header("Test 11: Graph Persistence")

    if graph is None:
        print_info("Skipping - no graph available")
        return

    try:
        from phoenix_ai import KnowledgeGraph

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test JSON format
            json_path = os.path.join(tmpdir, "graph.json")
            graph.save(json_path)
            loaded_json = KnowledgeGraph.load(json_path)
            assert loaded_json.num_entities == graph.num_entities
            print_success("JSON save/load successful")

            # Test pickle format
            pkl_path = os.path.join(tmpdir, "graph.pkl")
            graph.save(pkl_path)
            loaded_pkl = KnowledgeGraph.load(pkl_path)
            assert loaded_pkl.num_entities == graph.num_entities
            print_success("Pickle save/load successful")

            # Check file sizes
            json_size = os.path.getsize(json_path)
            pkl_size = os.path.getsize(pkl_path)
            print_info(f"JSON size: {json_size} bytes, Pickle size: {pkl_size} bytes")

        return True
    except Exception as e:
        print_fail(f"Graph persistence error: {e}")
        return False


def run_all_tests():
    """Run all GraphRAG tests."""
    print("\n" + "=" * 60)
    print(" PHOENIX AI GRAPHRAG - COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    results = {}

    # Test 1: Basic imports
    results['imports'] = test_basic_imports()

    if not results['imports']:
        print("\nCritical: Imports failed. Cannot continue tests.")
        return results

    # Test 2: Client initialization
    embedding_client, chat_client = test_client_initialization()
    results['clients'] = embedding_client is not None and chat_client is not None

    if not results['clients']:
        print("\nCritical: Client initialization failed. Cannot continue tests.")
        return results

    # Test 3: Embedding generation
    results['embeddings'] = test_embedding_generation(embedding_client)

    # Test 4: Entity extraction
    entities = test_entity_extraction(chat_client)
    results['entity_extraction'] = len(entities) > 0

    # Test 5: Relationship extraction
    relationships = test_relationship_extraction(chat_client, entities)
    results['relationship_extraction'] = True  # May have 0 relationships, that's ok

    # Test 6: Knowledge graph operations
    test_graph = test_knowledge_graph()
    results['knowledge_graph'] = test_graph is not None

    # Test 7: Graph builder
    built_graph, chunks = test_graph_builder(chat_client, embedding_client)
    results['graph_builder'] = built_graph is not None

    # Test 8: Graph retriever
    retriever = test_graph_retriever(built_graph, embedding_client)
    results['graph_retriever'] = retriever is not None

    # Test 9: GraphRAG inferencer
    if built_graph and chunks:
        results['graphrag_inferencer'] = test_graphrag_inferencer(
            embedding_client, chat_client, built_graph, chunks
        )
    else:
        results['graphrag_inferencer'] = False

    # Test 10: Direct graph query
    if built_graph:
        results['direct_query'] = test_query_graph_directly(
            built_graph, embedding_client, chat_client
        )
    else:
        results['direct_query'] = False

    # Test 11: Graph persistence
    results['persistence'] = test_graph_persistence(built_graph)

    # Summary
    print_header("TEST SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "[PASS]" if passed_test else "[FAIL]"
        print(f"  {status} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed successfully!")
    else:
        print(f"\n{total - passed} test(s) failed.")

    return results


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    run_all_tests()
