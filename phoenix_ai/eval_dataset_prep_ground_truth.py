import os
import re
import fitz  # PyMuPDF
import requests
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from openai import OpenAI, AzureOpenAI

class EvalDatasetGroundTruthGenerator:
    def __init__(self, chat_client):
        self.chat_client = chat_client

    def _pdf_to_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        return "".join(page.get_text() for page in doc)

    def _call_chat_client(self, user_input: str, max_tokens=3000):
        try:
            response = self.chat_client.chat(
                user_input=user_input,
                max_tokens=max_tokens
            )

            # Log the full response to inspect it
            print(f"Response from chat client: {response}")
            
            # Extract the JSON block using regex
            # match = re.search(r"```json\sto*(.*?)\s*```", response, re.DOTALL)
            match = re.search(r"\[.*?\]", response, re.DOTALL)
            if match:
                json_str = match.group(0).strip()  # Get the JSON part from the response and remove extra spaces
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
        
    def process_dataframe(self, df, text_column, prompt_template) -> pd.DataFrame:
        context_text = " ".join(df[text_column].tolist()).strip()
        prompt = prompt_template.format(context=context_text)

        qa_pairs = self._call_chat_client(prompt)
        if qa_pairs:
            return pd.DataFrame(qa_pairs)
        else:
            print("⚠️ No Q&A pairs returned.")
            # Return an empty DataFrame with expected columns to prevent downstream errors
            return pd.DataFrame(columns=["question", "ground truth"])