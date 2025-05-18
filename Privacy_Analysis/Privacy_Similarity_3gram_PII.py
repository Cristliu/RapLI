# Privacy_Similarity_3gram_PII.py

import json
import os
import sys
import glob
import pandas as pd
import tensorflow as tf  # Make sure to install the latest version: pip install tensorflow
import tensorflow_hub as hub  # Make sure to install the latest version: pip install tensorflow_hub
from tqdm import tqdm
import numpy as np
import gc
from spacy.lang.en import English

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from _01pii_detection.combined_detector import CombinedPIIDetector
import random
import torch
import time  

def set_random_seed(seed_value=42):
    """
    Set random seed to ensure reproducible results.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

# Disable all TensorFlow GPUs
def disable_tf_gpu():
    """
    Disable TensorFlow GPU usage to avoid conflicts with PyTorch GPU.
    """
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'

class USE:
    def __init__(self):
        with tf.device('/cpu:0'):
            self.encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def compute_sim(self, clean_texts, adv_texts):
        with tf.device('/cpu:0'):
            clean_embeddings = self.encoder(clean_texts)
            adv_embeddings = self.encoder(adv_texts)
            cosine_sim = tf.reduce_sum(clean_embeddings * adv_embeddings, axis=1)
            return cosine_sim.numpy()

def generate_ngrams(text, n):
    """
    Generate n-gram list.
    """
    words = text.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

def check_ngram_leakage(orig_text, perturbed_text, n=3):
    """
    Check n-gram leakage.
    """
    orig_ngrams = set(generate_ngrams(orig_text, n))
    perturbed_ngrams = set(generate_ngrams(perturbed_text, n))
    common_ngrams = orig_ngrams.intersection(perturbed_ngrams)
    leakage_proportion = len(common_ngrams) / len(orig_ngrams) if orig_ngrams else 0
    return len(common_ngrams), leakage_proportion

def check_pii_leakage(orig_text, perturbed_text, detector):
    """
    Check PII leakage.
    """
    orig_pii = detector.detect_pii(orig_text)
    perturbed_pii = detector.detect_pii(perturbed_text)
    orig_pii_words = set()
    perturbed_pii_words = set()
    perturbed_pii_words_list = []
    
    for entity in orig_pii:
        doc_orig = tokenizer(entity['text'])
        # Convert each token to lowercase and add to the set
        for token in doc_orig:
            orig_pii_words.add(token.text.lower())
    
    for entity in perturbed_pii:
        doc_perturbed = tokenizer(entity['text'])
        # Convert each token to lowercase
        tokens_lower = [token.text.lower() for token in doc_perturbed]
        perturbed_pii_words.update(tokens_lower)
        perturbed_pii_words_list.extend(tokens_lower)

    common_pii_words = orig_pii_words.intersection(perturbed_pii_words)

    len_common = len(common_pii_words)
    len_orig = len(orig_pii_words)
    len_perturbed = len(perturbed_pii_words)

    # Calculate recall
    recall = len_common / len_orig if len_orig > 0 else 0
    
    # Calculate precision
    precision = len_common / len_perturbed if len_perturbed > 0 else 0
    
    # Calculate F1 score (optional)
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return perturbed_pii_words_list, common_pii_words, len_common, precision, recall, f1_score


def save_detailed_metrics_results(
    df, 
    similarities, 
    ngram_leakage_props, 
    pii_leakage_R,
    save_root, 
    task, 
    dataset, 
    subdir, 
    base_filename
):
    """
    Save detailed metrics results for each row to a JSON file
    """
    # Create directory if it doesn't exist
    detailed_save_dir = os.path.join(save_root, task, dataset, subdir, "detailed_metrics")
    os.makedirs(detailed_save_dir, exist_ok=True)
    
    # Prepare the detailed results - text and requested metrics
    detailed_results = []
    for i, row in df.iterrows():
        detailed_results.append({
            "text": row["text"],
            "similarity": float(similarities[i]),
            "ngram_leakage_prop": float(ngram_leakage_props[i]),
            "pii_leakage_recall": float(pii_leakage_R[i])
        })
    
    # Save to JSON file
    detailed_json_path = os.path.join(detailed_save_dir, f"Detailed_Metrics_{base_filename}.json")
    try:
        with open(detailed_json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=4)
        print(f"Detailed metrics results saved to: {detailed_json_path}")
    except Exception as e:
        print(f"Error saving detailed metrics to {detailed_json_path}: {e}")

def process_csv_files():
    """
    Process all CSV files in specified paths, calculate similarity, n-gram leakage, PII leakage, and save results.
    """
    use = USE()
    detector = CombinedPIIDetector()
    
    # Define tasks and corresponding datasets
    tasks_datasets = {
        "topic_classification": ["ag_news"],
        "synthpai": ["synthpai"],
        "samsum": ["samsum"],
        "piidocs_classification": ["piidocs"]
    }
    
    # Define all possible subdirectories
    subdirectories = ["","Ours_Ablation" "Unsanitized"]
    
    # Define root directory for saving results
    save_root = 'Privacy_Results_FurtherAttackResults'
    
    # Iterate through each task and dataset
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
                
                for csv_path in tqdm(csv_paths, desc=f"Processing {task}/{dataset}/{subdir}"):
                    try:
                        df = pd.read_csv(csv_path)
                        #Process first 10 records
                        # df = df.head(10)
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")
                        continue

                    
                    # Initialize statistics lists
                    similarities = []
                    ngram_leakages_list = []
                    ngram_leakage_props = []
                    pii_leakages_cnt_list = []
                    pii_leakage_P = []
                    pii_leakage_R = []
                    pii_leakage_F1 = []
                    common_pii_words_list = []
                    all_perturbed_pii_words_list = []
                    
                    # Initialize processing time list
                    row_times = []
                    
                    # Iterate through each record
                    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating", leave=False):
                        text = str(row.get('text', ''))
                        perturbed = str(row.get('perturbed_sentence', ''))
                        
                        # Record start time
                        start_time = time.time()
                        
                        # Calculate similarity
                        sim = use.compute_sim([text], [perturbed])[0]
                        similarities.append(sim)
                        
                        # Check n-gram leakage
                        ngram_leakage, ngram_leakage_prop = check_ngram_leakage(text, perturbed)
                        ngram_leakages_list.append(ngram_leakage)
                        ngram_leakage_props.append(ngram_leakage_prop)
                        
                        # Check PII leakage
                        perturbed_pii_words_list, common_pii_words, len_common, precision, recall, f1_score = check_pii_leakage(text, perturbed, detector)
                        common_pii_words_list.append(list(common_pii_words))
                        pii_leakages_cnt_list.append(len_common)
                        pii_leakage_P.append(precision)
                        pii_leakage_R.append(recall)
                        pii_leakage_F1.append(f1_score)
                        all_perturbed_pii_words_list.append(perturbed_pii_words_list)
                        
                        # Record end time
                        end_time = time.time()
                        processing_time = end_time - start_time
                        row_times.append(processing_time)
                    
                    # Calculate statistics
                    avg_sim = float(np.mean(similarities)) if similarities else 0.0
                    total_ngram_leakage_cnt = int(np.sum(ngram_leakages_list))
                    avg_ngram_leakage_prop = float(np.mean(ngram_leakage_props)) if ngram_leakage_props else 0.0
                    total_pii_leakage_cnt = int(np.sum(pii_leakages_cnt_list))
                    avg_pii_leakage_P = float(np.mean(pii_leakage_P)) if pii_leakage_P else 0.0
                    avg_pii_leakage_R = float(np.mean(pii_leakage_R)) if pii_leakage_R else 0.0
                    avg_pii_leakage_F1 = float(np.mean(pii_leakage_F1)) if pii_leakage_F1 else 0.0
                    avg_proc_time = float(np.mean(row_times)) if row_times else 0.0
                    
                    # Print statistics
                    print(f"\n{csv_path} statistics:")
                    print(f"  - Average similarity: {avg_sim:.4f}")
                    print(f"  - Total n-gram leakage count: {total_ngram_leakage_cnt}")
                    print(f"  - Average n-gram leakage proportion: {avg_ngram_leakage_prop:.4f}")
                    print(f"  - Total PII leakage count: {total_pii_leakage_cnt}")
                    print(f"  - Average PII leakage precision: {avg_pii_leakage_P:.4f}")
                    print(f"  - Average PII leakage recall: {avg_pii_leakage_R:.4f}")
                    print(f"  - Average PII leakage F1 score: {avg_pii_leakage_F1:.4f}")
                    print(f"  - Average processing time: {avg_proc_time:.4f} seconds")
                    
                    
                    # Build save path
                    json_save_dir = os.path.join(save_root, task, dataset, subdir)
                    os.makedirs(json_save_dir, exist_ok=True)
                    
                    # Build JSON filename
                    base_filename = os.path.splitext(os.path.basename(csv_path))[0]
                    json_filename = f"Report_Sim_3gram_PII_{base_filename}.json"
                    json_path = os.path.join(json_save_dir, json_filename)
                    
                    # Build JSON content
                    result_json = {
                        "average_similarity": avg_sim,
                        "total_ngram_leakage": total_ngram_leakage_cnt,
                        "average_ngram_leakage_proportion": avg_ngram_leakage_prop,
                        "total_pii_leakage": total_pii_leakage_cnt,
                        "average_pii_leakage_Precision": avg_pii_leakage_P,
                        "average_pii_leakage_Recall": avg_pii_leakage_R,
                        "average_pii_leakage_F1": avg_pii_leakage_F1,
                        "average_processing_time_seconds": avg_proc_time
                    }
                    
                    # Save JSON file
                    try:
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(result_json, f, ensure_ascii=False, indent=4)
                        print(f"Evaluation results JSON saved to: {json_path}")
                    except Exception as e:
                        print(f"Error saving JSON {json_path}: {e}")
                    
                    # Add this:
                    try:
                        save_detailed_metrics_results(
                            df, 
                            similarities, 
                            ngram_leakage_props, 
                            pii_leakage_R,
                            save_root, 
                            task, 
                            dataset, 
                            subdir, 
                            base_filename
                        )
                    except Exception as e:
                        print(f"Error saving detailed metrics: {e}")



                    # Free memory
                    gc.collect()
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    set_random_seed(42)
    tokenizer = English()
    disable_tf_gpu()
    process_csv_files()