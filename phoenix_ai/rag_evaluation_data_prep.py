
import pandas as pd

class RagEvalDataPrep:
    def __init__(self, inferencer, system_prompt: str, index_type: str, index_path: str = None, index=None):
        self.inferencer = inferencer
        self.system_prompt = system_prompt
        self.index_type = index_type
        self.index_path = index_path
        self.index = index

    def _safe_generate_answer(self, question: str, top_k: int = 5, max_tokens: int = 256, mode: str = "standard") -> str:
        try:
            print(f"Generating answer for: {question}")
            kwargs = {
                "system_prompt": self.system_prompt,
                "question": question,
                "top_k": top_k,
                "max_tokens": max_tokens,
                "index_type": self.index_type,
                "mode": mode
            }

            if self.index_type == "local_index":
                kwargs["index_path"] = self.index_path
            elif self.index_type == "databricks_vector_index":
                kwargs["index"] = self.index
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")

            answer_df = self.inferencer.infer(**kwargs)
            return answer_df["answer"].iloc[0]

        except Exception as e:
            print(f"Error generating answer for question '{question}': {e}")
            return "Error generating answer"
        
    def run_rag(self, input_df: pd.DataFrame, top_k: int = 5, limit: int = None, mode: str = "standard") -> pd.DataFrame:
        # If a limit is set, trim the DataFrame
        if limit is not None:
            input_df = input_df.head(limit)

        # Ensure the column name is correct
        if 'question' not in input_df.columns:
            raise ValueError("Input DataFrame must contain a 'question' column.")

        # Create a new column for answers
        input_df['answer'] = input_df['question'].apply(lambda question: self._safe_generate_answer(question, top_k=top_k, mode=mode))

        print(f"\n✅ Results generated for {len(input_df)} questions.")
        return input_df
