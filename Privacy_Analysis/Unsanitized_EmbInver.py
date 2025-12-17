import torch
import random
import numpy as np
import logging
import os
import signal
import json
import csv
import time
from tqdm import tqdm
from spacy.lang.en import English
from nltk.corpus import stopwords
import string

import pandas as pd
from transformers import AutoTokenizer, AutoModel

def set_random_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

def main():
    set_random_seed(42)

    # Step 1: Load datasets
    tasks_datasets = {
        "topic_classification": ["ag_news"],
        "sentiment_analysis": ["SemEval"],
        "clinical_inference": ["mednli"],
        "piidocs_classification": ["piidocs"]
    }

    stop_words = set(stopwords.words('english'))
    punctuation_token = set(string.punctuation)
    stop_words = stop_words.union(punctuation_token)

    # Initialize BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    for task, datasets in tasks_datasets.items():
        for dataset in datasets:
            csv_path = os.path.join("NoiseResults", task, dataset, "Unsanitized", "Unsanitized.csv")
            if not os.path.exists(csv_path):
                print(f"{csv_path} not found, skipping.")
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Cannot read CSV file: {csv_path}, error: {e}")
                continue

            # Check column structure
            # Usually: text, ground_truth, perturbed_sentence
            # mednli: an additional hypothesis
            # But here we only need to analyze the text column
            if 'text' not in df.columns:
                print(f"'text' column not found in {csv_path}. Skipping.")
                continue

            embExpStop_rates = []

            # Step 2 & 3: BERT tokenization -> remove stopwords -> calculate remaining token length
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {task}/{dataset}"):
                raw_text = str(row['text'])
                # BERT tokenization
                tokens = tokenizer.tokenize(raw_text)
                total_len = len(tokens)
                if total_len == 0:
                    # If text is empty, record 0
                    embExpStop_rates.append(0.0)
                    continue

                # Remove stopwords
                filtered = [tok for tok in tokens if tok.lower() not in stop_words]
                remain_len = len(filtered)

                # Step 4: Calculate ratio
                embExpStop_rate = remain_len / total_len
                embExpStop_rates.append(embExpStop_rate)

            if len(embExpStop_rates) == 0:
                print(f"Cannot calculate embExpStop_rates, skipping save.")
                continue

            avg_embExpStop_rate = float(np.mean(embExpStop_rates))

            save_dir = os.path.join("NoiseResults_AttackResults", task, dataset, "Unsanitized")
            os.makedirs(save_dir, exist_ok=True)
            json_path = os.path.join(save_dir, "Unsanitized_EmbInver_1_attackrate.json")

            result = {
                "file": csv_path,
                "avg_embExpStop_rate": avg_embExpStop_rate
            }
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
                print(f"Saved to {json_path}")
            except Exception as e:
                print(f"Error writing JSON file: {e}")

if __name__ == "__main__":
    main()