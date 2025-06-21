import mlflow
import os
import pandas as pd
from phoenix_ai.vector_embedding_pipeline import VectorEmbedding
from phoenix_ai.rag_inference import RAGInferencer
from phoenix_ai.config_param import Param
from phoenix_ai.rag_evaluation_data_prep import RagEvalDataPrep
from phoenix_ai.rag_eval import RagEvaluator
from phoenix_ai.loaders import load_documents_to_dataframe
from phoenix_ai.utils import GenAIEmbeddingClient, GenAIChatClient

embedding_client = GenAIEmbeddingClient(
    provider="azure-openai",
    model="singtel-ai_coe-genai-models-pp-text-embedding-ada-002",
    api_key="bcf608e3f8cf45658a5d723761c6c3de",
    api_version="2024-06-01",
    azure_endpoint="https://openaidpmpoc.openai.azure.com/"
)

chat_client = GenAIChatClient(
    provider="azure-openai",
    model="singtel-ai_coe-genai-models-pp-gpt-4o",
    api_key="bcf608e3f8cf45658a5d723761c6c3de",
    api_version="2024-06-01",
    azure_endpoint="https://openaidpmpoc.openai.azure.com/"
)

eval_chat_client = GenAIChatClient(
    provider="databricks",
    model="databricks-claude-3-7-sonnet",
    base_url="https://adb-1086543832046777.17.azuredatabricks.net/serving-endpoints",
    api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

# Input data
df = load_documents_to_dataframe(folder_path="data/")       # DataFrame with a 'content' column
display(df.head())
qa_df = pd.read_csv("output/kb_eval_ground_truth.csv")    # Question-answer DataFrame for RAG evaluation
display(qa_df.head())

# Parameter combinations
chunk_settings = [(500, 50), (400, 40)]  # (chunk_size, overlap)
top_k_values = [3, 4, 6]
prompt_variants = {
    "Prompt_1": """You are an IT support assistant for Singtel employees. Your task is to provide accurate answers to IT-related questions based on the provided knowledge base context.\n\nGiven a context from Singtel's internal IT knowledge base and a specific question:\n1. First, carefully analyze the context to locate the exact information that addresses the question.\n2. Develop step-by-step reasoning to show how you arrived at the answer, considering:\n   - Relevant sections, tables, or bullet points in the context\n   - Specific procedures, troubleshooting steps, or support tiers mentioned\n   - Any conditional information (e.g., different processes for different user types)\n3. Provide a clear, concise answer that directly addresses the question.\n4. Include specific details like:\n   - Exact steps to follow for procedures\n   - Correct support teams to contact\n   - Relevant URLs or contact information\n   - Any prerequisites or important notes\n\nYour answer should be comprehensive enough to solve the employee's issue but concise enough to be immediately actionable. If the context contains partial or incomplete information, acknowledge this in your answer.\n\n""",
    "Prompt_2": "Given the fields `question`, produce the fields `answer`.",
    "Prompt_3": "You are an IT support assistant for Singtel employees. Think through the question step-by-step. Consider what information is being requested, relevant policies, standard procedures, and technical details that would help address the query. Then provide a clear, concise answer that directly addresses the question. Include specific timeframes, URLs, or procedural steps when relevant. Your response should be professional, accurate, and tailored to Singtel's internal systems and processes."
}

experiment_name = "/Users/praveen.govindaraj@singtel.com/VERA/VERA_HOT_FIX_Accuracy_KB_RAG/RAG_Evaluation_GridSearch"

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name)
print(f"MLflow experiment '{experiment_name}' set.")

# Loop through all combinations
for chunk_size, overlap in chunk_settings:
    # Create index
    vector = VectorEmbedding(embedding_client, chunk_size=chunk_size, overlap=overlap)
    index_path = f"index/kb_index_chunk{chunk_size}_overlap{overlap}.index"
    index_path, chunks = vector.generate_index(
        df=df,
        text_column="content",
        index_path=index_path,
        vector_index_type="local_index"
    )

    for prompt_name, system_prompt in prompt_variants.items():
        for top_k in top_k_values:
            run_name=f"{prompt_name}_topk_{top_k}_chunk{chunk_size}_overlap{overlap}"

            with mlflow.start_run(run_name=run_name):
                # Log tags (for filtering/searching in UI)
                mlflow.set_tag("prompt", prompt_name)
                mlflow.set_tag("top_k", top_k)
                mlflow.set_tag("chunk_size", chunk_size)
                mlflow.set_tag("overlap", overlap)

                mlflow.log_param("index_path", index_path)
                mlflow.log_param("chunk_size", chunk_size)
                mlflow.log_param("overlap", overlap)
                mlflow.log_param("top_k", top_k)
                mlflow.log_param("system_prompt", prompt_name)

                # Inference
                rag_inferencer = RAGInferencer(embedding_client, chat_client)

                # RAG Eval Data Prep
                rag_data = RagEvalDataPrep(
                    inferencer=rag_inferencer,
                    system_prompt=system_prompt,
                    index_path=index_path,
                    index_type="local_index"
                )

                result_df = rag_data.run_rag(input_df=qa_df, mode="standard", top_k=top_k)
                output_path = f"output/experimentation_pipeline/eval_dataset_{prompt_name}_topk_{top_k}_chunk{chunk_size}_overlap{overlap}.csv"
                result_df.to_csv(output_path, index=False)

                # Evaluation
                evaluator = RagEvaluator(eval_chat_client, experiment_name=experiment_name)
                df_eval, metrics = evaluator.evaluate(
                    input_df=result_df,
                    prompt=Param.get_evaluation_prompt(),
                    # max_rows=top_k
                    run_name=f"{prompt_name}_topk_{top_k}_chunk{chunk_size}_overlap{overlap}"
                )

                # Print the evaluation metrics
                print("📊 Metrics:")
                for k, v in metrics.items():
                    print(f"{k}: {v:.4f}")

                # Display the result DataFrame in the notebook (for Databricks)
                display(df_eval)

                df_eval.to_csv(output_path, index=False)

                mlflow.log_metrics(metrics)
                mlflow.log_artifact(output_path)

print("✅ All RAG evaluations completed for all combinations.")
