import os
import numpy as np
import faiss
import pickle
import pandas as pd

class VectorEmbedding:
    def __init__(self, embedding_client):
        self.client = embedding_client
        self.embedding_model = embedding_client.model

    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> list:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    def _generate_embeddings(self, chunks: list) -> list:
        result = self.client.generate_embedding(chunks)
        return result

    def generate_index(self, df: pd.DataFrame, text_column: str, index_path: str):
        all_text = "\n".join(df[text_column].dropna().astype(str).tolist())
        chunks = self._chunk_text(all_text)
        print(f" Loaded {len(chunks)} chunks from DataFrame.")

        embeddings = self._generate_embeddings(chunks)
        dim = len(embeddings[0])

        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))
        print(f" FAISS index size: {index.ntotal}")

        faiss.write_index(index, index_path)

        # Save chunks
        base_path = os.path.splitext(index_path)[0]
        chunk_path = base_path + "_chunks.pkl"

        with open(chunk_path, "wb") as f:
            pickle.dump(chunks, f)

        return index_path, chunks