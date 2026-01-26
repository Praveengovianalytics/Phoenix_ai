from typing import Dict, List, Optional

from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)


class MilvusVectorStore:
    def __init__(
        self,
        connection_args: Dict[str, str],
        collection_name: str,
        embedding_dim: int,
        id_field_name: str = "id",
        content_field_name: str = "content",
        vector_field_name: str = "embedding",
        metadata_fields: Optional[List[str]] = None,
        index_params: Optional[Dict[str, object]] = None,
        search_params: Optional[Dict[str, object]] = None,
        metric_type: str = "COSINE",
        consistency_level: str = "Session",
        drop_old: bool = False,
        connection_alias: str = "default",
    ):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.id_field_name = id_field_name
        self.content_field_name = content_field_name
        self.vector_field_name = vector_field_name
        self.metadata_fields = metadata_fields or []
        self.metric_type = metric_type
        self.consistency_level = consistency_level
        self.connection_alias = connection_alias
        self.index_params = index_params or {
            "metric_type": metric_type,
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 200},
        }
        self.search_params = search_params or {
            "metric_type": metric_type,
            "params": {"ef": 64},
        }

        connections.connect(alias=connection_alias, **connection_args)
        self.collection = self._get_or_create_collection(drop_old)
        self.collection.load()

    def _get_or_create_collection(self, drop_old: bool) -> Collection:
        if utility.has_collection(self.collection_name, using=self.connection_alias):
            if drop_old:
                utility.drop_collection(
                    self.collection_name, using=self.connection_alias
                )

        if not utility.has_collection(
            self.collection_name, using=self.connection_alias
        ):
            fields = [
                FieldSchema(
                    name=self.id_field_name,
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=False,
                    max_length=256,
                ),
                FieldSchema(
                    name=self.content_field_name,
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                ),
                FieldSchema(
                    name=self.vector_field_name,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.embedding_dim,
                ),
            ]

            for field_name in self.metadata_fields:
                fields.append(
                    FieldSchema(
                        name=field_name,
                        dtype=DataType.VARCHAR,
                        max_length=1024,
                    )
                )

            schema = CollectionSchema(
                fields=fields, description="Phoenix AI Milvus vector store"
            )
            collection = Collection(
                name=self.collection_name,
                schema=schema,
                using=self.connection_alias,
                consistency_level=self.consistency_level,
            )
            collection.create_index(
                field_name=self.vector_field_name,
                index_params=self.index_params,
            )
            return collection

        collection = Collection(self.collection_name, using=self.connection_alias)
        if not collection.indexes:
            collection.create_index(
                field_name=self.vector_field_name,
                index_params=self.index_params,
            )
        return collection

    def upsert_documents(self, documents: List[Dict[str, object]], batch_size: int = 200):
        if not documents:
            return

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            ids = [str(doc[self.id_field_name]) for doc in batch]
            contents = [str(doc[self.content_field_name]) for doc in batch]
            vectors = [doc[self.vector_field_name] for doc in batch]

            metadata_columns = []
            for field_name in self.metadata_fields:
                metadata_columns.append(
                    [
                        str(doc.get(field_name, ""))
                        for doc in batch
                    ]
                )

            data = [ids, contents, vectors] + metadata_columns
            self.collection.insert(data)

        self.collection.flush()

    def vector_search(self, query_vector: List[float], k: int = 3) -> List[str]:
        results = self.collection.search(
            data=[query_vector],
            anns_field=self.vector_field_name,
            param=self.search_params,
            limit=k,
            output_fields=[self.content_field_name],
            consistency_level=self.consistency_level,
        )
        hits = results[0]
        return [hit.entity.get(self.content_field_name) for hit in hits]
