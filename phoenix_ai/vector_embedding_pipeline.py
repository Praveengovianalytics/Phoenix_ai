import os
import pickle

import faiss
import numpy as np
import pandas as pd
from databricks.vector_search.client import VectorSearchClient

from .azure_ai_search import AzureAISearchVectorStore
from .milvus_vector_store import MilvusVectorStore


class VectorEmbedding:
    def __init__(self, embedding_client, chunk_size, overlap):
        self.client = embedding_client
        self.embedding_model = embedding_client.model
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _chunk_text(self, text: str) -> list:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.overlap
        return chunks

    def _generate_embeddings(self, chunks: list) -> list:
        return self.client.generate_embedding(chunks)

    def generate_faiss_index(self, df: pd.DataFrame, text_column: str, index_path: str):
        # Validate that the directory of index_path exists
        dir_path = os.path.dirname(index_path)
        if dir_path and not os.path.exists(dir_path):
            raise FileNotFoundError(
                f"The directory '{dir_path}' does not exist. Please provide a valid index_path."
            )

        all_text = "\n".join(df[text_column].dropna().astype(str).tolist())
        chunks = self._chunk_text(all_text)
        embeddings = self._generate_embeddings(chunks)
        dim = len(embeddings[0])

        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))

        faiss.write_index(index, index_path)

        # Save chunks
        base_path = os.path.splitext(index_path)[0]
        chunk_path = base_path + "_chunks.pkl"

        with open(chunk_path, "wb") as f:
            pickle.dump(chunks, f)

        print(f"FAISS index saved with {len(chunks)} chunks at {index_path}")

        return index_path, chunks

    def batched_upsert(self, index, records, batch_size=200):
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            try:
                index.upsert(batch)
                print(f"Upserted batch {i // batch_size + 1}: {len(batch)} records")
            except Exception as e:
                print(f"❌ Failed batch {i // batch_size + 1}: {e}")
                raise

    def generate_databricks_index(
        self,
        df: pd.DataFrame,
        content_column: str,
        catalog: str,
        schema: str,
        endpoint_name: str,
        embedding_dim: int,
        index_name: str,
    ):
        # Filter and reset index
        df = df[df[content_column].astype(str).str.strip() != ""].reset_index(drop=True)
        df["id"] = df.index

        # Generate embeddings
        contents = df[content_column].astype(str).tolist()
        contents = [c for c in contents if c.strip() != ""]

        print(f"Generating embeddings for {len(contents)} texts")
        print(f"Sample content: {contents[:3]}")

        embeddings = self.client.generate_embedding(contents)
        df["embedding"] = embeddings

        # Create vector search index
        index_name = f"{catalog}.{schema}.{endpoint_name}_{index_name}"
        vs_client = VectorSearchClient()

        # Prepare records for upsert
        records = (
            df[["id", content_column, "embedding"]]
            .rename(columns={content_column: "content"})
            .to_dict(orient="records")
        )

        index_schema = {"id": "int", "content": "string", "embedding": "array<float>"}

        # Create the index if it doesn't exist
        try:
            vs_client.create_direct_access_index(
                endpoint_name=endpoint_name,
                index_name=index_name,
                primary_key="id",
                embedding_dimension=embedding_dim,
                embedding_vector_column="embedding",
                schema=index_schema,
            )
        except Exception as e:
            # If index exists, get the existing one
            if "RESOURCE_ALREADY_EXISTS" in str(e):
                print(f"Index {index_name} already exists, retrieving it instead.")
            else:
                raise e

        try:
            index = vs_client.get_index(
                endpoint_name=endpoint_name, index_name=index_name
            )
            index.wait_until_ready()
            self.batched_upsert(index, records)
            print(f"Upserted {len(records)} records into index {index_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to upsert into Databricks index: {e}")

    def generate_index(
        self,
        df: pd.DataFrame,
        text_column: str,
        index_path: str,
        vector_index_type: str = "local_index",
        **kwargs,
    ):
        if vector_index_type == "local_index":
            if index_path is None:
                raise ValueError("index_path must be specified for local_index type")
            return self.generate_faiss_index(df, text_column, index_path)
        elif vector_index_type == "databricks_vector_index":
            required_args = [
                "catalog",
                "schema",
                "endpoint_name",
                "embedding_dim",
                "index_name",
            ]
            missing_args = [arg for arg in required_args if arg not in kwargs]
            if missing_args:
                raise ValueError(
                    f"Missing arguments for Databricks index: {missing_args}"
                )
            return self.generate_databricks_index(
                df=df,
                content_column=text_column,
                catalog=kwargs["catalog"],
                schema=kwargs["schema"],
                endpoint_name=kwargs["endpoint_name"],
                embedding_dim=kwargs["embedding_dim"],
                index_name=kwargs["index_name"],
            )
        elif vector_index_type == "azure_ai_search_vector_index":
            required_args = ["search_service_endpoint", "index_name", "embedding_dim"]
            missing_args = [arg for arg in required_args if arg not in kwargs]
            if missing_args:
                raise ValueError(
                    f"Missing arguments for Azure AI Search index: {missing_args}"
                )

            df = df[df[text_column].astype(str).str.strip() != ""].reset_index(
                drop=True
            )
            id_field_name = kwargs.get("id_field_name", "id")
            content_field_name = kwargs.get("content_field_name", "content")
            vector_field_name = kwargs.get("vector_field_name", "embedding")
            title_field_name = kwargs.get("title_field_name")
            metadata_fields = kwargs.get("metadata_fields")

            df[id_field_name] = df.index.astype(str)

            contents = df[text_column].astype(str).tolist()
            embeddings = self.client.generate_embedding(contents)

            documents = []
            for row_index, content, embedding in zip(df.index, contents, embeddings):
                row = df.loc[row_index]
                document = {
                    id_field_name: row[id_field_name],
                    content_field_name: content,
                    vector_field_name: embedding,
                }
                if title_field_name and title_field_name in df.columns:
                    document[title_field_name] = str(row[title_field_name])
                if metadata_fields:
                    for field_name in metadata_fields:
                        if field_name in df.columns:
                            document[field_name] = str(row[field_name])
                documents.append(document)

            store = AzureAISearchVectorStore(
                search_service_endpoint=kwargs["search_service_endpoint"],
                index_name=kwargs["index_name"],
                embedding_dim=kwargs["embedding_dim"],
                credential=kwargs.get("credential"),
                search_api_key=kwargs.get("search_api_key"),
                id_field_name=id_field_name,
                content_field_name=content_field_name,
                vector_field_name=vector_field_name,
                title_field_name=title_field_name,
                metadata_fields=metadata_fields,
                vector_search_profile_name=kwargs.get(
                    "vector_search_profile_name", "default-hnsw"
                ),
                hnsw_algorithm_configuration_name=kwargs.get(
                    "hnsw_algorithm_configuration_name", "hnsw-config"
                ),
                hnsw_metric=kwargs.get("hnsw_metric", "cosine"),
                hnsw_m=kwargs.get("hnsw_m", 4),
                hnsw_ef_construction=kwargs.get("hnsw_ef_construction", 200),
                hnsw_ef_search=kwargs.get("hnsw_ef_search", 300),
                update_index=kwargs.get("update_index", False),
            )
            store.upsert_documents(documents)
            return store
        elif vector_index_type == "milvus_vector_index":
            required_args = ["connection_args", "collection_name", "embedding_dim"]
            missing_args = [arg for arg in required_args if arg not in kwargs]
            if missing_args:
                raise ValueError(
                    f"Missing arguments for Milvus index: {missing_args}"
                )

            df = df[df[text_column].astype(str).str.strip() != ""].reset_index(
                drop=True
            )
            id_field_name = kwargs.get("id_field_name", "id")
            content_field_name = kwargs.get("content_field_name", "content")
            vector_field_name = kwargs.get("vector_field_name", "embedding")
            metadata_fields = kwargs.get("metadata_fields")

            df[id_field_name] = df.index.astype(str)

            contents = df[text_column].astype(str).tolist()
            embeddings = self.client.generate_embedding(contents)

            documents = []
            for row_index, content, embedding in zip(df.index, contents, embeddings):
                row = df.loc[row_index]
                document = {
                    id_field_name: row[id_field_name],
                    content_field_name: content,
                    vector_field_name: embedding,
                }
                if metadata_fields:
                    for field_name in metadata_fields:
                        if field_name in df.columns:
                            document[field_name] = str(row[field_name])
                documents.append(document)

            store = MilvusVectorStore(
                connection_args=kwargs["connection_args"],
                collection_name=kwargs["collection_name"],
                embedding_dim=kwargs["embedding_dim"],
                id_field_name=id_field_name,
                content_field_name=content_field_name,
                vector_field_name=vector_field_name,
                metadata_fields=metadata_fields,
                index_params=kwargs.get("index_params"),
                search_params=kwargs.get("search_params"),
                metric_type=kwargs.get("metric_type", "COSINE"),
                consistency_level=kwargs.get("consistency_level", "Session"),
                drop_old=kwargs.get("drop_old", False),
                connection_alias=kwargs.get("connection_alias", "default"),
            )
            store.upsert_documents(documents)
            return store
        else:
            raise ValueError(f"Unsupported vector_index_type: {vector_index_type}")
