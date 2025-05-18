import os
import glob
import json
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from baseline.rantext.local_llm.local_llm_denoise import LocalLLMDenoise
from evaluation.evaluate_local_llm_llama8b import evaluate_results_llm

def set_random_seed(seed_value=42):
    """Set random seed to ensure reproducible results."""
    random.seed(seed_value)
    np.random.seed(seed_value)

def extract_label_from_result(final_result):
    """
    Same logic as in main_blackllm.py to extract label from final_result.
    """
    try:
        # Improved regex that allows spaces after numbers
        match = re.search(r"\{\s*[\"']label[\"']\s*:\s*(\d+)\s*\}", final_result)
        if match:
            label = int(match.group(1))  # Extract number and convert to integer
        else:
            print(f"\nfinal_result_llama3-8b_response: {final_result}")
            label = 100  # 100 represents processing error
            raise ValueError("No valid label found in the output. Setting prediction to 100.")
    except Exception as e:
        print(f"Error processing sentence: {e}")
        label = 100  # 100 represents processing error

    return label

def run_local_llm(local_llm, model, prompt):
    """Call local_llm.denoise_output and return results while measuring processing time."""
    start_time = time.time()
    result = local_llm.denoise_output(model=model, prompt=prompt)
    processing_time = time.time() - start_time
    return result, processing_time

def process_csv_file(csv_file, local_llm, local_llm_model, task, dataset, subdir):
    """Process a single CSV file and save results with subdirectory name included."""
    try:
        df = pd.read_csv(csv_file)
        df = df.head(10)  # For testing convenience, only take first 10 records
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return

    if 'text' not in df.columns or 'perturbed_sentence' not in df.columns or 'gpt_response' not in df.columns:
        print(f"Missing required columns in CSV, skipping file: {csv_file}")
        return

    original_texts = df['text'].astype(str).tolist()
    perturbed_sentences = df['perturbed_sentence'].astype(str).tolist()
    gpt_responses = df['gpt_response'].astype(str).tolist()


    # If ground_truth column is available for evaluation
    if 'ground_truth' in df.columns:
        ground_truths = df['ground_truth'].tolist()
    else:
        ground_truths = [None] * len(df)

    all_results = []
    row_times = []
    predictions = [None] * len(df)

        # Submit all tasks first
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_info = {}
        for i, (text, p_sentence, gpt_resp, hypo) in enumerate(zip(original_texts, perturbed_sentences, gpt_responses, hypotheses)):
            if task == "clinical_inference" and hypo is not None:
                local_input = (
                    f"'Original sentence': {text};\n"
                    f"'Disturbed sentence': {p_sentence};\n"
                    f"'GPT's analysis and classification': {gpt_resp};\n"
                    f"'Hypothesis': {hypo}"
                )
            else:
                local_input = (
                    f"'Original sentence': {text};\n"
                    f"'Disturbed sentence': {p_sentence};\n"
                    f"'GPT's analysis and classification': {gpt_resp}"
                )

            future = executor.submit(run_local_llm, local_llm, local_llm_model, local_input)
            future_to_info[future] = (i,)  # Only need to save the index

        # Then wait for all results to complete, using a progress bar
        for future in tqdm(as_completed(future_to_info), total=len(future_to_info), desc="Processing records"):
            index = future_to_info[future][0]
            try:
                final_result, processing_time = future.result()
                row_times.append(processing_time)
                
                label = extract_label_from_result(final_result)
                result_entry = {
                    "text": original_texts[index],
                    "perturbed_sentence": perturbed_sentences[index],
                    "gpt_response": gpt_responses[index],
                    "final_result": final_result,
                    "label": label,
                }

                if ground_truths[index] is not None:
                    result_entry["ground_truth"] = ground_truths[index]

                all_results.append(result_entry)
                predictions[index] = label
                
            except Exception as e:
                print(f"Error processing row {index} in {csv_file}: {e}")
                final_result = "Error"
                row_times.append(0.0)
                predictions[index] = 100  # Error code

    avg_time = np.mean(row_times) if row_times else 0.0

    # Build save path (with subdirectory)
    save_dir = os.path.join('evaluation_results', f"{task}_{dataset}_gpt-3.5-turbo-ca_llama8b_denoise", subdir)
    os.makedirs(save_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    subdir_suffix = f"_{subdir}" if subdir else ""

    json_filename = f"LocalDenoise{subdir_suffix}_{base_filename}.json"
    csv_filename = f"LocalDenoise{subdir_suffix}_{base_filename}.csv"
    json_save_path = os.path.join(save_dir, json_filename)
    csv_save_path = os.path.join(save_dir, csv_filename)

    # Save results to JSON file
    try:
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print(f"Evaluation results JSON saved to: {json_save_path}")
    except Exception as e:
        print(f"Error saving JSON {json_save_path}: {e}")

    # Save results to CSV file
    try:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(csv_save_path, index=False, encoding="utf-8")
        print(f"Evaluation results CSV saved to: {csv_save_path}")
    except Exception as e:
        print(f"Error saving CSV {csv_save_path}: {e}")

    # If ground_truth is available, can perform evaluation
    if any(ground_truths):
        evaluate_results_llm(task, dataset, base_filename, ground_truths, predictions, local_llm_model, avg_time)

def main():
    set_random_seed(42)

    local_llm_model = "llama3:latest" # deepseek-r1:14b
    base_url = "http://localhost:3000/ollama/api/chat"
    api_key = "***"  # Replace with actual key

    tasks = {
        "topic_classification": ["ag_news"],
        # Can continue to add other tasks
    }

    subdirectories = ["", "Ours_Ablation"]#

    print("\nStarting to process tasks:")
    for task, datasets in tasks.items():
        # Instantiate local LLM
        local_llm = LocalLLMDenoise(task, base_url=base_url, api_key=api_key)
        for dataset in datasets:
            print(f"Processing dataset: {dataset}")
            
            for subdir in subdirectories:
                # For "" subdirectory, build pattern without subdirectory; for "Ours_Ablation", build pattern with subdirectory
                if subdir:
                    csv_pattern = os.path.join('evaluation_results', f"{task}_{dataset}_gpt-3.5-turbo-ca/{subdir}", '*.csv')
                else:
                    csv_pattern = os.path.join('evaluation_results', f"{task}_{dataset}_gpt-3.5-turbo-ca", '*.csv')

                csv_paths = glob.glob(csv_pattern)
                if not csv_paths:
                    print(f"Warning: No CSV files found under path {csv_pattern}, skipping.")
                    continue

                # Only process epsilon_1.0 files (if needed)
                csv_paths = [p for p in csv_paths if "epsilon_1.0" in p]

                for csv_file in tqdm(csv_paths, desc=f"Processing CSV in {task}/{dataset}/{subdir}"):
                    process_csv_file(csv_file, local_llm, local_llm_model, task, dataset, subdir)

if __name__ == "__main__":
    main()