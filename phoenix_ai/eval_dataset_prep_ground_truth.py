import json
import os
import re
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF
import pandas as pd
import requests
from openai import AzureOpenAI, OpenAI


class EvalDatasetGroundTruthGenerator:
    def __init__(self, chat_client):
        self.chat_client = chat_client

    def _pdf_to_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        return "".join(page.get_text() for page in doc)

    def _call_chat_client(self, user_input: str, max_tokens=3000):
        try:
            response = self.chat_client.chat(
                user_input=user_input, max_tokens=max_tokens
            )

            # Log the full response to inspect it
            print(f"Response from chat client: {response}")

            # Extract the JSON block using regex
            # match = re.search(r"```json\sto*(.*?)\s*```", response, re.DOTALL)
            match = re.search(r"\[.*?\]", response, re.DOTALL)
            if match:
                json_str = match.group(
                    0
                ).strip()  # Get the JSON part from the response and remove extra spaces
                # Remove any trailing commas before closing brackets/braces
                json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
                try:
                    # Parse the JSON string
                    qa_pairs = json.loads(json_str)
                    return qa_pairs
                except json.JSONDecodeError:
                    print("Error: The extracted JSON is invalid.")
                    return None
            else:
                print("Error: JSON block not found in response.")
                return None

        except Exception as e:
            print(f"Error during chat response: {e}")
            return None

    def count_batches(
        self, df: pd.DataFrame, text_column: str = "content", max_chars: int = 10000
    ) -> int:
        current_length = 0
        batch_count = 0
        current_batch = []

        for row_text in df[text_column]:
            row_text = row_text.strip()
            if not row_text:
                continue
            row_len = len(row_text)

            if current_length + row_len > max_chars and current_batch:
                batch_count += 1
                current_batch = [row_text]
                current_length = row_len
            else:
                current_batch.append(row_text)
                current_length += row_len

        if current_batch:
            batch_count += 1

        return batch_count

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "content",
        max_chars: int = 10000,
        prompt_template: str = "",
        max_total_pairs=None,
    ) -> pd.DataFrame:
        # Smart chunking based on character limit
        all_qa_pairs = []
        current_batch = []
        current_length = 0

        batch_count = self.count_batches(df, text_column="content", max_chars=10000)
        print(f"📦 Estimated total batches to send: {batch_count}")

        for row_text in df[text_column]:
            row_text = row_text.strip()
            if not row_text:
                continue
            row_len = len(row_text)

            if current_length + row_len > max_chars and current_batch:
                context_text = " ".join(current_batch)
                prompt = prompt_template.format(context=context_text)
                qa_pairs = self._call_chat_client(prompt)

                # Check limit before adding
                if max_total_pairs is not None:
                    remaining = max_total_pairs - len(all_qa_pairs)
                    if remaining <= 0:
                        break
                    qa_pairs = qa_pairs[:remaining]

                all_qa_pairs.extend(qa_pairs)

                if max_total_pairs is not None and len(all_qa_pairs) >= max_total_pairs:
                    break

                current_batch = [row_text]
                current_length = row_len
            else:
                current_batch.append(row_text)
                current_length += row_len

        # Final batch
        if current_batch and (
            max_total_pairs is None or len(all_qa_pairs) < max_total_pairs
        ):
            context_text = " ".join(current_batch)
            prompt = prompt_template.format(context=context_text)
            qa_pairs = self._call_chat_client(prompt)

            if max_total_pairs is not None:
                remaining = max_total_pairs - len(all_qa_pairs)
                qa_pairs = qa_pairs[:remaining]

            all_qa_pairs.extend(qa_pairs)

        if not all_qa_pairs:
            print("⚠️ No Q&A pairs returned.")
        else:
            # Normalize keys (e.g., fix inconsistent naming of 'ground_truth' vs 'ground truth')
            normalized_qa_pairs = []
            for pair in all_qa_pairs:
                normalized_pair = {**pair}
                if "ground_truth" in normalized_pair:
                    normalized_pair["ground truth"] = normalized_pair.pop(
                        "ground_truth"
                    )
                normalized_qa_pairs.append(normalized_pair)
                print(f"✅ Total Q&A pairs generated: {len(normalized_qa_pairs)}")

        return pd.DataFrame(normalized_qa_pairs)
