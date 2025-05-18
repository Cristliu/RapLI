import os
import json
import pandas as pd
import re
import glob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from tqdm import tqdm
import torch
from rouge_score import rouge_scorer
from bert_score import score as bertscore

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def extract_summary_from_response(response):
    """
    Extract summary information from LLM response
    """
    match = re.search(r'\{\s*[\'"]summary[\'"]\s*:\s*[\'"]([^\'"]*)[\'"]', response)
    if match: 
        return match.group(1)
    else:
        return ""

def get_rouge1_scores(summaries, references):
    """Calculate ROUGE-1 scores according to evaluation.py standard"""
    scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    
    for i in range(len(summaries)):
        ref = references[i]
        summ = summaries[i]
        
        # Split into sentences and join with newlines (consistent with evaluation.py) ==> This splitting operation likely improves results
        ref = "\n".join(sent_tokenize(ref))
        summ = "\n".join(sent_tokenize(summ))
        
        # Calculate ROUGE-1 scores
        rouge_scores = scorer.score(ref, summ)
        r1 = 100 * rouge_scores["rouge1"].fmeasure  # Convert to percentage
        scores.append(r1)
        
    return np.array(scores)

def get_rouge2_scores(summaries, references):
    """Calculate ROUGE-2 scores according to evaluation.py standard"""
    scores = []
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    
    for i in range(len(summaries)):
        ref = references[i]
        summ = summaries[i]
        
        # Split into sentences and join with newlines
        ref = "\n".join(sent_tokenize(ref))
        summ = "\n".join(sent_tokenize(summ))
        
        # Calculate ROUGE-2 scores
        rouge_scores = scorer.score(ref, summ)
        r2 = 100 * rouge_scores["rouge2"].fmeasure
        scores.append(r2)
        
    return np.array(scores)

def get_rougel_scores(summaries, references):
    """Calculate ROUGE-L scores according to evaluation.py standard"""
    scores = []
    scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
    
    for i in range(len(summaries)):
        ref = references[i]
        summ = summaries[i]
        
        # Split into sentences and join with newlines
        ref = "\n".join(sent_tokenize(ref))
        summ = "\n".join(sent_tokenize(summ))
        
        # Calculate ROUGE-L scores
        rouge_scores = scorer.score(ref, summ)
        rl = 100 * rouge_scores["rougeLsum"].fmeasure
        scores.append(rl)
        
    return np.array(scores)

def get_meanrouge_scores(summaries, references):
    """Calculate average of ROUGE-1, ROUGE-2 and ROUGE-L scores"""
    scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    
    for i in range(len(summaries)):
        ref = references[i]
        summ = summaries[i]
        
        # Split into sentences and join with newlines
        ref = "\n".join(sent_tokenize(ref))
        summ = "\n".join(sent_tokenize(summ))
        
        # Calculate all ROUGE scores
        rouge_scores = scorer.score(ref, summ)
        r1 = 100 * rouge_scores["rouge1"].fmeasure
        r2 = 100 * rouge_scores["rouge2"].fmeasure
        rl = 100 * rouge_scores["rougeLsum"].fmeasure
        
        # Calculate average
        mean_rouge = (r1 + r2 + rl) / 3
        scores.append(mean_rouge)
        
    return np.array(scores)

def get_bertscore_scores(summaries, references):
    """Calculate BERTScore scores according to evaluation.py standard"""
    # BERTScore needs complete lists
    p, r, f1 = bertscore(summaries, references, lang='en', verbose=True)
    
    # Convert to numpy array and to percentage
    scores = 100 * f1.detach().cpu().numpy()
    return scores

def process_json_file(json_file, result_dir):
    """Process a single JSON file (keeping original reading and storage logic)"""
    # Extract filename for output results
    rel_path = os.path.relpath(json_file, result_dir)
    base_name = os.path.basename(json_file).replace("RespAppend_", "").replace(".json", "")
    
    # If file is in a subfolder, include the subfolder name in the output path
    output_subdir = os.path.dirname(rel_path)
    
    # Read results data
    with open(json_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Extract summaries and ground truth texts
    summaries = []
    ground_truths = []
    
    for item in results:
        response = item.get("gpt_response", "")
        ground_truth = item.get("ground_truth", "")
        
        summary = extract_summary_from_response(response)
        
        summaries.append(summary)
        ground_truths.append(ground_truth)
    
    # Filter out empty summaries and ground truths
    valid_summaries = []
    valid_ground_truths = []
    for gt, summary in zip(ground_truths, summaries):
        if summary and gt:
            valid_summaries.append(summary)
            valid_ground_truths.append(gt)
    
    # If there are no valid samples, return zero values
    if not valid_summaries:
        print(f"No valid summaries found in {json_file}")
        eval_results = {
            'rouge-1-f': 0.0,
            'rouge-2-f': 0.0,
            'rouge-l-f': 0.0,
            'mean-rouge': 0.0,
            'bertscore': 0.0
        }
    else:
        # Calculate evaluation metrics
        try:
            r1_scores = get_rouge1_scores(valid_summaries, valid_ground_truths)
            r2_scores = get_rouge2_scores(valid_summaries, valid_ground_truths)
            rl_scores = get_rougel_scores(valid_summaries, valid_ground_truths)
            mean_rouge_scores = get_meanrouge_scores(valid_summaries, valid_ground_truths)
            
            # Calculate BERTScore (if possible)
            try:
                bertscore_scores = get_bertscore_scores(valid_summaries, valid_ground_truths)
                bertscore_mean = np.mean(bertscore_scores)
            except Exception as e:
                print(f"Error calculating BERTScore: {e}")
                bertscore_mean = 0.0
            
            # Summarize results
            eval_results = {
                'rouge-1-f': float(np.mean(r1_scores)),
                'rouge-2-f': float(np.mean(r2_scores)),
                'rouge-l-f': float(np.mean(rl_scores)),
                'mean-rouge': float(np.mean(mean_rouge_scores)),
                'bertscore': float(bertscore_mean)
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            eval_results = {
                'rouge-1-f': 0.0,
                'rouge-2-f': 0.0,
                'rouge-l-f': 0.0,
                'mean-rouge': 0.0,
                'bertscore': 0.0
            }
    
    # Output evaluation results
    print(f"Evaluation Results for {rel_path}:")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Save evaluation results to corresponding subdirectory
    eval_dir = os.path.join(result_dir, "Eval_Results", output_subdir)
    os.makedirs(eval_dir, exist_ok=True)
    
    json_save_path = os.path.join(eval_dir, f"ExtendedEvalResults_{base_name}.json")
    
    try:
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=4)
        print(f"Extended results saved to {json_save_path}\n")
        return True
    except Exception as e:
        print(f"Error saving JSON {json_save_path}: {e}")
        return False

def evaluate_samsum_results(result_dir):
    """
    Evaluate SamSum summary results, including all subdirectories (keeping original logic)
    """
    # Recursively find all JSON files starting with RespAppend_
    all_json_files = []
    for root, _, files in os.walk(result_dir):
        for file in files:
            if file.startswith("RespAppend_") and file.endswith(".json"):
                all_json_files.append(os.path.join(root, file))
    
    if not all_json_files:
        print(f"No RespAppend_*.json files found in {result_dir} or its subdirectories.")
        return
    
    processed = 0
    for json_file in tqdm(all_json_files, desc="Processing files"):
        if process_json_file(json_file, result_dir):
            processed += 1
    
    print(f"Completed processing {processed} files in {result_dir}")

def main():
    # Main evaluation directory
    base_dir = "evaluation_results"
    
    # Find all SamSum result directories
    samsum_dirs = glob.glob(os.path.join(base_dir, "samsum_samsum_gpt-3.5-turbo-ca"))
    
    if not samsum_dirs:
        print("No SamSum evaluation results found.")
        return
    
    for result_dir in samsum_dirs:
        print(f"Processing results in {result_dir}")
        evaluate_samsum_results(result_dir)

if __name__ == "__main__":
    main()