import os
import numpy as np
import faiss
import pickle
import pandas as pd
from databricks.vector_search.client import VectorSearchClient

class VectorEmbedding:
    def __init__(self, embedding_client,chunk_size,overlap):
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
            batch = records[i:i + batch_size]
            try:
                index.upsert(batch)
                print(f"Upserted batch {i // batch_size + 1}: {len(batch)} records")
            except Exception as e:
                print(f"❌ Failed batch {i // batch_size + 1}: {e}")
                raise

    def generate_databricks_index(self, df: pd.DataFrame, content_column: str,
                              catalog: str, schema: str,
                              endpoint_name: str, embedding_dim: int, index_name: str):
        # Filter and reset index
        df = df[df[content_column].astype(str).str.strip() != ''].reset_index(drop=True)
        df["id"] = df.index

        # Generate embeddings
        contents = df[content_column].astype(str).tolist()
        contents = [c for c in contents if c.strip() != '']

        print(f"Generating embeddings for {len(contents)} texts")
        print(f"Sample content: {contents[:3]}")

        embeddings = self.client.generate_embedding(contents)
        df["embedding"] = embeddings

        # Create vector search index
        index_name = f"{catalog}.{schema}.{endpoint_name}_{index_name}"
        vs_client = VectorSearchClient()

        # Prepare records for upsert
        records = df[["id", content_column, "embedding"]].rename(
            columns={content_column: "content"}
        ).to_dict(orient="records")

        index_schema = {
            "id": "int",
            "content": "string",
            "embedding": "array<float>"
        }

        # Create the index if it doesn't exist
        try:
            vs_client.create_direct_access_index(
                endpoint_name=endpoint_name,
                index_name=index_name,
                primary_key="id",
                embedding_dimension=embedding_dim,
                embedding_vector_column="embedding",
                schema=index_schema
            )
        except Exception as e:
            # If index exists, get the existing one
            if "RESOURCE_ALREADY_EXISTS" in str(e):
                print(f"Index {index_name} already exists, retrieving it instead.")
            else:
                raise e

        try:
            index = vs_client.get_index(endpoint_name=endpoint_name, index_name=index_name)
            index.wait_until_ready()
            self.batched_upsert(index, records)
            print(f"Upserted {len(records)} records into index {index_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to upsert into Databricks index: {e}")
    
    def generate_index(self, df: pd.DataFrame, text_column: str, index_path: str, vector_index_type: str = 'local_index', **kwargs):
        if vector_index_type == 'local_index':
            if index_path is None:
                raise ValueError("index_path must be specified for local_index type")
            return self.generate_faiss_index(df, text_column, index_path)
        elif vector_index_type == 'databricks_vector_index':
            required_args = ['catalog', 'schema', 'endpoint_name', 'embedding_dim', 'index_name']
            missing_args = [arg for arg in required_args if arg not in kwargs]
            if missing_args:
                raise ValueError(f"Missing arguments for Databricks index: {missing_args}")
            return self.generate_databricks_index(
                df=df,
                content_column=text_column,
                catalog=kwargs['catalog'],
                schema=kwargs['schema'],
                endpoint_name=kwargs['endpoint_name'],
                embedding_dim=kwargs['embedding_dim'],
                index_name=kwargs['index_name']
            )
        else:
            raise ValueError(f"Unsupported vector_index_type: {vector_index_type}")
