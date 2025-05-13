import pandas as pd

class RagEvalDataPrep:
    def __init__(self, inferencer, system_prompt: str, index_path: str):
        self.inferencer = inferencer
        self.system_prompt = system_prompt
        self.index_path = index_path

    def _safe_generate_answer(self, question: str, top_k: int = 5, max_tokens: int = 256) -> str:
        try:
            print(f"Generating answer for: {question}")
            answer = self.inferencer.infer(
                system_prompt=self.system_prompt,
                index_path=self.index_path,
                question=question,
                top_k=top_k,
                max_tokens=max_tokens
            )
            #print(f"Answer: {answer['answer'].iloc[0]}\n{'-' * 40}")
            return answer["answer"].iloc[0]
        except Exception as e:
            print(f"Error generating answer for question '{question}': {e}")
            return "Error generating answer"
        
    def run_rag(self, input_df: pd.DataFrame, top_k: int = 5, limit: int = None) -> pd.DataFrame:
        # If a limit is set, trim the DataFrame
        if limit is not None:
            input_df = input_df.head(limit)

        # Ensure the column name is correct
        if 'question' not in input_df.columns:
            raise ValueError("Input DataFrame must contain a 'question' column.")

        # Create a new column for answers
        input_df['answer'] = input_df['question'].apply(lambda question: self._safe_generate_answer(question, top_k=top_k))

        print(f"\n✅ Results generated for {len(input_df)} questions.")
        return input_df
