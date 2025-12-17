# main_blackllm.py

import re
import json
import os
import sys
import glob
import pandas as pd
from _04black_llm.black_llm_interface import BlackLLMInterface
from evaluation.evaluate import evaluate_results
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
import time
import numpy as np

# Set random seed
def set_random_seed(seed_value=42):
    """
    Set random seed to ensure reproducible results.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)

class BlackLLMEvaluator:
    def __init__(self, api_key):
        """
        Initialize BlackLLMInterface.
        """
        self.api_key = api_key

    def query_black_llm(self, black_llm, prompt, model):
        """
        Helper function to query black-box LLM and record processing time.
        """
        start_time = time.time()
        response = black_llm.query(prompt, model)
        end_time = time.time()
        processing_time = end_time - start_time
        return response, processing_time

    def extract_label_from_response(self, response):
        """
        Extract label number from response
        """
        match = re.search(r"\{\s*[\"']label[\"']\s*:\s*(\d+)\s*\}", response)
        if match:
            return int(match.group(1))
        return 100  # 100 represents processing error

    def evaluate_folder(self, task, dataset, subdir, csv_paths, black_llm_model_name):
        """
        Evaluate all CSV files in a single folder.
        """
        if not csv_paths:
            print(f"Warning: No CSV files found under path {csv_paths}, skipping.")
            return
        
        for csv_file in tqdm(csv_paths, desc=f"Processing CSV files for {task}/{dataset}/{subdir}"):
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue

            # Extract necessary data
            original_texts = df['text'].astype(str).tolist()
            perturbed_sentences = df['perturbed_sentence'].astype(str).tolist()
            ground_truths = df['ground_truth'].tolist()



            # Instantiate BlackLLMInterface
            black_llm = BlackLLMInterface(task, dataset, self.api_key)

            predictions = [None] * len(perturbed_sentences)
            all_results = []
            row_times = []
            all_responses = [None] * len(perturbed_sentences)


            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=100) as executor:
                # Adjust input format based on task type
                prompts = perturbed_sentences

                # Submit tasks to thread pool and record future to index mapping
                future_to_index = {
                    executor.submit(self.query_black_llm, black_llm, prompt, black_llm_model_name): index
                    for index, prompt in enumerate(prompts)
                }

                for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing records"):
                    index = future_to_index[future]
                    response, processing_time = future.result()
                    all_responses[index] = response  # Store response using original index
                    
                    # Process response based on task type
                    if task == "samsum":
                        # For samsum task, no need to extract label, just save response
                        label = None
                    else:
                        # For other tasks, extract label
                        label = self.extract_label_from_response(response)
                    
                    # Record processing time
                    row_times.append(processing_time)

                    # Build result entry
                    result_entry = {
                        "text": original_texts[index],
                        "ground_truth": ground_truths[index],
                        "perturbed_sentence": perturbed_sentences[index],
                        "gpt_response": response,
                    }
                    
                    # For non-summary tasks, add final prediction label
                    if task != "samsum":
                        result_entry["final_result"] = label
                        predictions[index] = label

                    all_results.append(result_entry)

            # Calculate average processing time
            avg_time = np.mean(row_times) if row_times else 0.0

            # Get current csv filename
            csv_file_name = os.path.splitext(os.path.basename(csv_file))[0]

            # Build save path
            json_save_dir = os.path.join('evaluation_results', f"{task}_{dataset}_{black_llm_model_name}", subdir)
            os.makedirs(json_save_dir, exist_ok=True)

            # Build JSON and CSV filenames
            base_filename = os.path.splitext(os.path.basename(csv_file))[0]
            json_filename = f"RespAppend_{base_filename}.json"
            csv_save_path = os.path.join(json_save_dir, f"RespAppend_{base_filename}.csv")
            json_save_path = os.path.join(json_save_dir, json_filename)

            # Save results to JSON file
            try:
                with open(json_save_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=4)
                print(f"GPT evaluation results JSON saved to: {json_save_path}")
            except Exception as e:
                print(f"Error saving JSON {json_save_path}: {e}")

            # Save results to CSV file
            try:
                df_results = pd.DataFrame(all_results)
                df_results.to_csv(csv_save_path, index=False, encoding="utf-8")
                print(f"GPT evaluation results CSV saved to: {csv_save_path}")
            except Exception as e:
                print(f"Error saving CSV {csv_save_path}: {e}")

            # Evaluate results
            try:
                if task == "samsum":
                    # For samsum task, pass responses for ROUGE evaluation
                    evaluate_results(task, dataset, csv_file_name, ground_truths, None, 
                                    black_llm_model_name, avg_time, responses=all_responses)
                else:
                    # For other tasks, pass ground_truths for evaluation
                    evaluate_results(task, dataset, csv_file_name, ground_truths, 
                                    predictions, black_llm_model_name, avg_time)
            except Exception as e:
                print(f"Error during result evaluation: {e}")

def main():
    set_random_seed(42)
    
    # Define tasks and corresponding datasets
    tasks_datasets = {
        "topic_classification": ["ag_news"],
        "piidocs_classification": ["piidocs"],
        "samsum": ["samsum"],
    }
    
    # Define all possible subdirectories
    subdirectories = ["", "Unsanitized", "Ours_Ablation"]#""

    # Define black-box LLM model name
    black_llm_model_name = "gpt-3.5-turbo-ca"  # Can be adjusted as needed
    #Paid: gpt-4-turbo-ca // deepseek-reasoner is deepseek-r1
    #Free: gpt-3.5-turbo-ca, gpt-4o-mini, gpt-4o-ca
    
    # Replace with your API key
    api_key = 'sk-***'  # Please ensure secure storage and management of your API key
    
    # Instantiate BlackLLMEvaluator
    evaluator = BlackLLMEvaluator(api_key)
    
    # Iterate through each task, dataset, and subdirectory
    for task, datasets in tasks_datasets.items():
        for dataset in datasets:
            for subdir in subdirectories:
                # Build path for reading CSV files
                if subdir:
                    csv_pattern = os.path.join('NoiseResults', task, dataset, subdir, '*.csv')
                else:
                    csv_pattern = os.path.join('NoiseResults', task, dataset, '*.csv')
  
                csv_paths = glob.glob(csv_pattern)
                 
                
                if not csv_paths:
                    print(f"Warning: No CSV files found under path {csv_pattern}, skipping.")
                    continue
                
                evaluator.evaluate_folder(task, dataset, subdir, csv_paths, black_llm_model_name)

if __name__ == "__main__":
    main()