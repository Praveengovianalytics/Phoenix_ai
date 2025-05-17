import pandas as pd
import mlflow
import re
import os
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from sklearn.metrics import precision_score, recall_score, f1_score
# from config_param import Param

class RagEvaluator:
    def __init__(self, chat_client, experiment_name: str,):
        self.chat_client = chat_client
        self.smoothing_function = SmoothingFunction().method4

        # Setup MLflow
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment '{experiment_name}' set.")

    def _extract_llm_scores(self, response_text):
        scores = {
            "answer_relevance": 0.0,
            "accuracy": 0.0,
            "completeness": 0.0,
            "clarity": 0.0,
            "semantic_similarity": 0,
            "explanation": response_text.strip()
        }
        try:
            scores["answer_relevance"] = float(re.search(r'Answer Relevance:\s*([0-1](?:\.\d+)?)', response_text).group(1))
            scores["accuracy"] = float(re.search(r'Accuracy:\s*([0-1](?:\.\d+)?)', response_text).group(1))
            scores["completeness"] = float(re.search(r'Completeness:\s*([0-1](?:\.\d+)?)', response_text).group(1))
            scores["clarity"] = float(re.search(r'Clarity:\s*([0-1](?:\.\d+)?)', response_text).group(1))
            scores["semantic_similarity"] = float(re.search(r'Semantic Similarity Score:\s*(\d{1,3})%', response_text).group(1))
        except Exception as e:
            print(f"Error extracting scores: {e}")
        return scores


    def _get_llm_judgment(self, prompt, ground_truth, generated_answer):
        prompt = prompt.format(ground_truth=ground_truth, generated_answer=generated_answer)
        result_text = self.chat_client.chat(prompt)
        print(f"Response from LLM: {result_text}")
        return self._extract_llm_scores(result_text)

    def evaluate(self, input_df: pd.DataFrame, prompt: str = None, max_rows: int = None):
        if max_rows:
            input_df = input_df.head(max_rows)
        if prompt is None:
            prompt = prompt

        bleu_scores, semantic_similarities = [], []
        answer_relevance, accuracies, completenesses, clarities, explanations = [], [], [], [], []

        binary_true, binary_pred = [], []

        # Iterate over each row in the input DataFrame
        for idx, row in input_df.iterrows():
            ground_truth = str(row['ground truth']).strip()
            generated_answer = str(row['answer']).strip()

            reference = [ground_truth.split()]
            candidate = generated_answer.split()
            bleu = sentence_bleu(reference, candidate, smoothing_function=self.smoothing_function)
            bleu_scores.append(bleu)

            scores = self._get_llm_judgment(prompt, ground_truth, generated_answer)
            answer_relevance.append(scores["answer_relevance"])
            semantic_similarities.append(scores["semantic_similarity"])
            accuracies.append(scores["accuracy"])
            completenesses.append(scores["completeness"])
            clarities.append(scores["clarity"])
            explanations.append(scores["explanation"])

            binary_true.append(1)
            binary_pred.append(1 if scores["semantic_similarity"] and scores["semantic_similarity"] >= 80 else 0)

            input_df.at[idx, 'BLEU Score'] = bleu
            input_df.at[idx, 'Answer Relevance'] = scores["answer_relevance"]
            input_df.at[idx, 'Accuracy'] = scores["accuracy"]
            input_df.at[idx, 'Completeness'] = scores["completeness"]
            input_df.at[idx, 'Clarity'] = scores["clarity"]
            input_df.at[idx, 'Semantic Similarity'] = scores["semantic_similarity"]
            input_df.at[idx, 'Explanation'] = scores["explanation"]

        # Calculate metrics
        precision = precision_score(binary_true, binary_pred)
        recall = recall_score(binary_true, binary_pred)
        f1 = f1_score(binary_true, binary_pred)

        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        avg_relevance = sum(answer_relevance) / len(answer_relevance)
        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_completeness = sum(completenesses) / len(completenesses)
        avg_clarity = sum(clarities) / len(clarities)
        avg_similarity = sum(semantic_similarities) / len(semantic_similarities)

        metrics = {
            "avg_bleu": avg_bleu,
            "avg_answer_relevance": avg_relevance,
            "avg_accuracy": avg_accuracy,
            "avg_completeness": avg_completeness,
            "avg_clarity": avg_clarity,
            "avg_semantic_similarity": avg_similarity,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        # Log metrics to MLflow
        with mlflow.start_run():
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            mlflow.log_params({
                "evaluation_size": len(input_df),
                "total_questions": len(input_df)
            })

            # Optionally, you can log the evaluation DataFrame as an artifact
            # mlflow.log_artifact(input_df, artifact_path="evaluation_results")

        print("✅ Evaluation complete. Metrics logged to MLflow.")
        return input_df, metrics
