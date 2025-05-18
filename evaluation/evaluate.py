# evaluate.py
import json
import os
import re
from sklearn.metrics import accuracy_score, f1_score
from rouge import Rouge
import nltk
import numpy as np

# Try to load NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def extract_summary_from_response(response):
    """
    Extract summary information from LLM response
    """
    # Try to find summary in JSON dictionary format
    match = re.search(r'\{\s*[\'"]summary[\'"]\s*:\s*[\'"]([^\'"]*)[\'"]', response)
    if match:
        return match.group(1)
    else:
        # If no JSON format summary can be found, return empty string
        return ""

def evaluate_results(task, define_dataset, csv_file_name, ground_truths, predictions, black_llm_model_name, avg_time, responses=None):
    results = {}  
    
    if task in ["topic_classification", "piidocs_classification"]:
        # Calculate basic metrics
        accuracy = accuracy_score(ground_truths, predictions)
        f1 = f1_score(ground_truths, predictions, average='weighted')
        results["accuracy"] = accuracy
        results["f1_score"] = f1
        results["avg_time"] = avg_time
        
        
    # Modify the samsum part in evaluate_results function
    elif task == "samsum":
        # Evaluate SamSum task - dialogue summarization
        if responses is None:
            print("Error: SamSum task requires responses parameter")
            results = {
                'rouge-1-recall': 0.0, 'rouge-1-precision': 0.0, 'rouge-1-f': 0.0,
                'rouge-2-recall': 0.0, 'rouge-2-precision': 0.0, 'rouge-2-f': 0.0,
                'rouge-l-recall': 0.0, 'rouge-l-precision': 0.0, 'rouge-l-f': 0.0,      
                "avg_time": avg_time
            }
        else:
            rouge = Rouge()
            # Extract predicted summaries
            summaries = [extract_summary_from_response(response) for response in responses]
            
            # Calculate ROUGE metrics
            rouge_scores = {
                'rouge-1-recall': 0.0, 'rouge-1-precision': 0.0, 'rouge-1-f': 0.0,
                'rouge-2-recall': 0.0, 'rouge-2-precision': 0.0, 'rouge-2-f': 0.0,
                'rouge-l-recall': 0.0, 'rouge-l-precision': 0.0, 'rouge-l-f': 0.0
            }
            valid_count = 0
            
            for gt, summary in zip(ground_truths, summaries):
                # Check if summary is not empty
                if summary and gt:
                    try:
                        # Calculate ROUGE scores for current sample
                        scores = rouge.get_scores(summary, gt)[0]
                        # Save Recall, Precision and F-score
                        rouge_scores['rouge-1-recall'] += scores['rouge-1']['r']
                        rouge_scores['rouge-1-precision'] += scores['rouge-1']['p']
                        rouge_scores['rouge-1-f'] += scores['rouge-1']['f']
                        
                        rouge_scores['rouge-2-recall'] += scores['rouge-2']['r']
                        rouge_scores['rouge-2-precision'] += scores['rouge-2']['p']
                        rouge_scores['rouge-2-f'] += scores['rouge-2']['f']
                        
                        rouge_scores['rouge-l-recall'] += scores['rouge-l']['r']
                        rouge_scores['rouge-l-precision'] += scores['rouge-l']['p']
                        rouge_scores['rouge-l-f'] += scores['rouge-l']['f']
                        
                        
                        valid_count += 1
                    except Exception as e:
                        print(f"Error calculating metrics for: '{summary}' against '{gt}': {e}")
            
            # Calculate averages
            results = {}
            print(f"Valid samples count: {valid_count}")
            if valid_count > 0:
                # Calculate averages for all Rouge metrics
                for key in rouge_scores:
                    results[key] = rouge_scores[key] / valid_count
                
                # Maintain compatibility with original format while adding new detailed metrics
                results["rouge-1"] = results["rouge-1-f"]  # Keep original rouge-1 key for compatibility
                results["rouge-2"] = results["rouge-2-f"]  # Keep original rouge-2 key for compatibility
                results["rouge-l"] = results["rouge-l-f"]  # Keep original rouge-l key for compatibility
                
                results["avg_time"] = avg_time
            else:
                results = {
                    'rouge-1-recall': 0.0, 'rouge-1-precision': 0.0, 'rouge-1-f': 0.0,
                    'rouge-2-recall': 0.0, 'rouge-2-precision': 0.0, 'rouge-2-f': 0.0,
                    'rouge-l-recall': 0.0, 'rouge-l-precision': 0.0, 'rouge-l-f': 0.0,
                    'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0,
                    "avg_time": avg_time
                }

    # Output evaluation results
    print(f"Evaluation Results for {task}_{define_dataset}_{black_llm_model_name}_{csv_file_name}:")
    for metric, value in results.items():
        print(f"{metric}: {value}") 

    # Build save path
    save_path = os.path.join("evaluation_results", f"{task}_{define_dataset}_{black_llm_model_name}", "Eval_Results")
    os.makedirs(save_path, exist_ok=True)

    # Build JSON filename
    json_filename = f"EvalResults_{csv_file_name}.json"
    json_save_path = os.path.join(save_path, json_filename)

    # Save evaluation results to JSON file
    try:
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {json_save_path}\n\n")
    except Exception as e:
        print(f"Error saving JSON {json_save_path}: {e}")