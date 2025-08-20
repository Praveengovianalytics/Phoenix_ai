<p align="center">
  <img src="assets/phoenix_logo.png" alt="Phoenix AI" width="280" />
</p>

# 🔥 phoenix_ai

**phoenix_ai** is a modular Python library designed for GenAI tasks like-------:

- 🔍 Vector embedding with FAISS
- 🤖 RAG Inference (Standard / Hybrid / HyDE)
- 📄 Ground truth Q&A generation from documents
- 🧪 Answer evaluation using BLEU + LLM-as-a-Judge (ChatGPT or Claude)
- 📊 MLflow logging of evaluation metrics

Supports:  
🧠 OpenAI | ☁️ Azure OpenAI | 💼 Databricks Model Serving | 🏠 Ollama (local) | 🔓 Sentence Transformers (local embeddings)

---

## 📦 Installation

```bash
git clone https://github.com/your-org/phoenix_ai.git
cd phoenix_ai
poetry install
```

---

## ⚙️ 1. Configure Embedding & Chat Clients

Supports `openai`, `azure-openai`, `databricks`, `ollama` (local), and `sentence-transformer` (local embeddings).

### ▶️ OpenAI
```python
from phoenix_ai.utils import GenAIEmbeddingClient, GenAIChatClient

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
```

### ☁️ Azure OpenAI
```python
embedding_client = GenAIEmbeddingClient(
    provider="azure-openai",
    model="text-embedding-3-large",
    api_key="your-azure-key",
    api_version="2024-06-01",
    azure_endpoint="https://<your-endpoint>.openai.azure.com"
)

chat_client = GenAIChatClient(
    provider="azure-openai",
    model="gpt-4o",
    api_key="your-azure-key",
    api_version="2024-06-01",
    azure_endpoint="https://<your-endpoint>.openai.azure.com"
)
```

### 💼 Databricks Model Serving
```python
embedding_client = GenAIEmbeddingClient(
    provider="databricks",
    model="bge_large_en_v1_5",
    base_url="https://<your-databricks-url>",
    api_key="your-databricks-token"
)

chat_client = GenAIChatClient(
    provider="databricks",
    model="databricks-claude-3-7-sonnet",
    base_url="https://<your-databricks-url>",
    api_key="your-databricks-token"
)
```

### 🏠 Ollama (Local LLM)
```bash
ollama serve &
ollama pull llama3.1
```
```python
from phoenix_ai.utils import GenAIChatClient

chat_client = GenAIChatClient(
    provider="ollama",
    model="llama3.1",  # or another local model
    # base_url defaults to http://localhost:11434/v1
)
print(chat_client.chat("Hello Phoenix!"))
```

### 🔓 Sentence Transformers (Free Local Embeddings)
```bash
pip install "sentence-transformers>=2.6.1,<3.0.0"
ollama pull nomic-embed-text  # optional if you also want local embeddings via Ollama
```
```python
from phoenix_ai.utils import GenAIEmbeddingClient

embedding_client = GenAIEmbeddingClient(
    provider="sentence-transformer",
    model="all-MiniLM-L6-v2",  # or any sentence-transformers model
    device="cpu"               # or "cuda"
)
embeddings = embedding_client.generate_embedding(["hello", "world"]) 
```

---

## 📂 2. Load and Process Documents
```python
from phoenix_ai.loaders import load_and_process_single_document

df = load_and_process_single_document(folder_path="data/", filename="policy_doc.pdf")
```

## 📌 3. Generate FAISS Vector Index
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

## 💬 4. Perform RAG Inference (Standard, Hybrid, HyDE)
```python
from phoenix_ai.rag_inference import RAGInferencer
from phoenix_ai.param import Param

rag_inferencer = RAGInferencer(embedding_client, chat_client)
response_df = rag_inferencer.infer(
    system_prompt=Param.get_rag_prompt(),
    index_path="output/policy_doc.index",
    question="What is the purpose of the company Group Data Classification Policy?",
    mode="standard",  # or "hybrid", "hyde"
    top_k=5
)
```

## 🧪 5. Generate Ground Truth Q&A from Document
```python
from phoenix_ai.eval_dataset_prep_ground_truth import EvalDatasetGroundTruthGenerator

generator = EvalDatasetGroundTruthGenerator(chat_client)
qa_df = generator.process_dataframe(
    df=df,
    text_column="content",
    prompt_template=Param.get_ground_truth_prompt(),
    max_total_pairs=50
)
qa_df.to_csv("output/eval_dataset_ground_truth.csv", index=False)
```

## 🔁 6. Apply RAG to Ground Truth Questions
```python
from phoenix_ai.rag_evaluation_data_prep import RagEvalDataPrep

rag_data = RagEvalDataPrep(
    inferencer=rag_inferencer,
    system_prompt=Param.get_rag_prompt(),
    index_type="local_index",
    index_path="output/policy_doc.index"
)

result_df = rag_data.run_rag(input_df=qa_df, limit=5)
result_df.to_csv("output/eval_dataset_rag_output.csv", index=False)
```

## 📊 7. Evaluate RAG Output with LLM-as-a-Judge
```python
from phoenix_ai.rag_eval import RagEvaluator

evaluator = RagEvaluator(chat_client, experiment_name="/Users/yourname/LLM_Answer_Evaluation")
df_input = result_df

df_eval, metrics = evaluator.evaluate(
    input_df=df_input,
    prompt=Param.get_evaluation_prompt(),
    max_rows=5
)

df_eval.to_csv("output/eval_dataset_rag_eval.csv", index=False)

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
```
