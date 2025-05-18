import json
import os
import sys
import pandas as pd
import numpy as np
import torch
import random
import ast
import glob
import time
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from torch.cuda.amp import autocast
import gc
from spacy.lang.en import English
from concurrent.futures import ThreadPoolExecutor, as_completed

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from _01pii_detection.combined_detector import CombinedPIIDetector

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_random_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

def parse_pii_words(pii_str):
    if isinstance(pii_str, list):
        return pii_str
    if pd.isna(pii_str) or pii_str.strip() == '[]':
        return []
    try:
        return ast.literal_eval(pii_str)
    except:
        return []

def extract_pii(text, detector):
    orig_pii = detector.detect_pii(text)
    results = []
    for entity in orig_pii:
        doc_orig = tokenizer(entity['text'])
        tokens_lower = [token.text.lower() for token in doc_orig]
        results.extend(tokens_lower)
    return results

# Parallel PII extraction + parsing
def parallel_extract_pii(df, input_column, output_column, detector, max_workers=50):
    def _worker(idx, text):
        try:
            extracted = extract_pii(text, detector)
            parsed = parse_pii_words(extracted)
            return (idx, parsed, None)
        except Exception as e:
            return (idx, [], e)

    results = []
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, row in df.iterrows():
            val = row[input_column]
            future = executor.submit(_worker, idx, val)
            futures_map[future] = idx

        with tqdm(total=len(df), desc=f"Parallel extracting {output_column}") as pbar:
            for future in as_completed(futures_map):
                idx = futures_map[future]
                _idx, parsed_val, err = future.result()
                if err:
                    pass
                results.append((_idx, parsed_val))
                pbar.update(1)

    results.sort(key=lambda x: x[0])
    for idx, val in results:
        df.at[idx, output_column] = val
    return df

###################################
# Cache & Embeddings
###################################
embedding_cache = {}
perturbed_embedding_cache = {}

def precompute_original_embeddings(df, model, batch_size=32):
    all_original_pii = [pii for sublist in df['original_pii'] if sublist for pii in sublist]
    unique_original_pii = list(set(all_original_pii))

    for i in tqdm(range(0, len(unique_original_pii), batch_size), desc="Batch encoding original_pii"):
        batch = unique_original_pii[i:i+batch_size]
        batch_to_encode = [word for word in batch if word not in embedding_cache]
        if batch_to_encode:
            with autocast():
                embeddings = model.encode(
                    batch_to_encode,
                    convert_to_tensor=True,
                    device=device,
                    show_progress_bar=False,
                    batch_size=batch_size
                )
            for word, emb in zip(batch_to_encode, embeddings):
                embedding_cache[word] = emb.cpu()

    df['original_pii_embeddings'] = df['original_pii'].apply(
        lambda pii_list: [embedding_cache[word] for word in pii_list if word in embedding_cache]
        if pii_list else []
    )
    return df

def precompute_perturbed_embeddings(df, model, device, batch_size=32):
    all_perturbed_pii = []
    for row in df['all_perturbed_pii_words']:
        if row:
            all_perturbed_pii.extend(row)
    unique_perturbed_pii = list(set(all_perturbed_pii))

    for i in tqdm(range(0, len(unique_perturbed_pii), batch_size), desc="Batch encoding perturbed_pii"):
        batch = unique_perturbed_pii[i:i+batch_size]
        batch_to_encode = [w for w in batch if w not in perturbed_embedding_cache]
        if not batch_to_encode:
            continue
        with autocast():
            emb_batch = model.encode(
                batch_to_encode,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False,
                batch_size=batch_size
            )
        for w, emb in zip(batch_to_encode, emb_batch):
            perturbed_embedding_cache[w] = emb.cpu()

    def get_perturbed_embeddings(row_tokens):
        return [perturbed_embedding_cache[t] for t in row_tokens if t in perturbed_embedding_cache]

    df['perturbed_pii_embeddings'] = df['all_perturbed_pii_words'].apply(get_perturbed_embeddings)
    return df


# Add this function to save detailed match results
def save_detailed_match_results(df, save_root, task, dataset, subdir, base_filename):
    """
    Save detailed match results for each row to a JSON file
    """
    # Create directory if it doesn't exist
    detailed_save_dir = os.path.join(save_root, task, dataset, subdir, "detailed_results")
    os.makedirs(detailed_save_dir, exist_ok=True)
    
    # Prepare the detailed results - just text and match status
    detailed_results = []
    for idx, row in df.iterrows():
        detailed_results.append({
            "text": row["text"],
            "is_match_success": int(row["is_match_success"])  # Convert boolean to int (0/1)
        })
    
    # Save to JSON file
    detailed_json_path = os.path.join(detailed_save_dir, f"Detailed_PII_match_{base_filename}.json")
    try:
        with open(detailed_json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=4)
        print(f"Detailed match results saved to: {detailed_json_path}")
    except Exception as e:
        print(f"Error saving detailed results to {detailed_json_path}: {e}")


###################################
# Simplified single row matching logic
###################################
def simplified_find_matches(perturbed_embeddings, original_embeddings, threshold=0.7, match_percentage=0.6):
    """
    Only compare original_pii_embeddings and perturbed_pii_embeddings for the current row.
    Returns True if similarity meets threshold, otherwise False.
    """
    start_time = time.time()

    if not perturbed_embeddings or not original_embeddings:
        return False, [], [], 0

    try:
        pe = torch.stack(perturbed_embeddings, dim=0)  # (num_perturbed, embedding_dim)
        oe = torch.stack(original_embeddings, dim=0)   # (num_original, embedding_dim)
    except Exception as e:
        print(f"Error stacking embeddings: {e}")
        return False, [], [], 0

    # Calculate similarity matrix (num_perturbed, num_original)
    similarities = util.cos_sim(pe, oe)
    matched_original = (similarities >= threshold).any(dim=0)  # Check for each original token if any perturbed exceeds threshold
    matched_count = matched_original.sum().item()
    match_ratio = matched_count / len(original_embeddings)

    # Determine match if ratio >= match_percentage, or if matched tokens > 4 and match_ratio >= 0.5
    if match_ratio >= match_percentage or (matched_count > 4 and match_ratio >= 0.5):
        return True, [], [], time.time() - start_time  # Empty lists when text/original_pii content not needed
    else:
        return False, [], [], time.time() - start_time

def match_single_row(row):
    """
    Read original_pii_embeddings / perturbed_pii_embeddings from this row
    Do a simplified comparison, without traversing the entire table.
    """
    perturbed_embeddings = row.get('perturbed_pii_embeddings', [])
    original_embeddings = row.get('original_pii_embeddings', [])
    return simplified_find_matches(
        perturbed_embeddings,
        original_embeddings,
        threshold=0.7,
        match_percentage=0.6
    )

###################################
# Report saving
###################################
def save_json_report(save_root, task, dataset, subdir, base_filename, report_data):
    json_save_dir = os.path.join(save_root, task, dataset, subdir)
    os.makedirs(json_save_dir, exist_ok=True)
    json_filename = f"Report_PII_match_{base_filename}.json"
    json_path = os.path.join(json_save_dir, json_filename)
    
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=4)
        print(f"Evaluation results JSON saved to: {json_path}\n")
    except Exception as e:
        print(f"Error saving JSON {json_path}: {e}")

def save_total_report(all_reports, save_root):
    total_json_path = os.path.join(save_root, 'total_report.json')
    try:
        with open(total_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_reports, f, ensure_ascii=False, indent=4)
        print(f"\nTotal evaluation results JSON saved to: {total_json_path}")
    except Exception as e:
        print(f"Error saving total JSON {total_json_path}: {e}")

###################################
# Main process
###################################
def process_csv_files():
    detector = CombinedPIIDetector()
    
    tasks_datasets = {
        "topic_classification": ["ag_news"],
        "synthpai": ["synthpai"],
        "samsum": ["samsum"],
        "piidocs_classification": ["piidocs"]
    }
    
    subdirectories = ["","Ours_Ablation" "Unsanitized", ]
    save_root = 'Privacy_Results_FurtherAttackResults'
    all_reports = []
    
    model_name = 'all-mpnet-base-v2'
    model = SentenceTransformer(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for task, datasets in tasks_datasets.items():
        for dataset in datasets:
            for subdir in subdirectories:
                if subdir:
                    csv_pattern = os.path.join('NoiseResults', task, dataset, subdir, '*.csv')
                else:
                    csv_pattern = os.path.join('NoiseResults', task, dataset, '*.csv')
                
                csv_paths = glob.glob(csv_pattern)


                if not csv_paths:
                    print(f"Warning: No CSV files found under path {csv_pattern}, skipping.")
                    continue
                
                for csv_path in tqdm(csv_paths, desc=f"Processing {task}/{dataset}/{subdir}"):
                    print(f"\nProcessing file: {csv_path}")
                    try:
                        df = pd.read_csv(csv_path)
                    except Exception as e:
                        print(f"Failed to read CSV file {csv_path}: {e}")
                        all_reports.append({"file": csv_path, "error": f"Failed to read CSV: {e}"})
                        continue
                    
                    # Parallel extraction of "all_perturbed_pii_words"
                    try:
                        df['all_perturbed_pii_words'] = [[] for _ in range(len(df))]
                        df = parallel_extract_pii(
                            df,
                            input_column='perturbed_sentence',
                            output_column='all_perturbed_pii_words',
                            detector=detector,
                            max_workers=50
                        )
                    except Exception as e:
                        print(f"Error during perturbed PII extraction for file {csv_path}: {e}")
                        all_reports.append({"file": csv_path, "error": f"Perturbed PII extraction error: {e}"})
                        continue

                    # Parallel extraction of "original_pii"
                    try:
                        df['original_pii'] = [[] for _ in range(len(df))]
                        df = parallel_extract_pii(
                            df,
                            input_column='text',
                            output_column='original_pii',
                            detector=detector,
                            max_workers=50
                        )
                    except Exception as e:
                        print(f"Error during original PII extraction for file {csv_path}: {e}")
                        all_reports.append({"file": csv_path, "error": f"Original PII extraction error: {e}"})
                        continue
                    
                    # Precompute embeddings
                    try:
                        df = precompute_original_embeddings(df, model)
                        df = precompute_perturbed_embeddings(df, model, device)
                    except Exception as e:
                        print(f"Error during embedding computation for file {csv_path}: {e}")
                        all_reports.append({"file": csv_path, "error": f"Embedding computation error: {e}"})
                        continue
                    
                    # Parallel matching (simplified one-to-one comparison)
                    try:
                        futures = {}
                        match_results = []
                        with ThreadPoolExecutor(max_workers=50) as executor:
                            with tqdm(total=len(df), desc="Applying matching function") as pbar:
                                for idx, row in df.iterrows():
                                    future = executor.submit(match_single_row, row)
                                    futures[future] = idx
                                for future in as_completed(futures):
                                    idx = futures[future]
                                    try:
                                        is_match, _, _, match_time = future.result()
                                        match_results.append([idx, is_match, match_time])
                                    except Exception:
                                        match_results.append([idx, False, 0])
                                    pbar.update(1)

                        match_results.sort(key=lambda x: x[0])
                        match_results_df = pd.DataFrame(match_results, columns=[
                            'index','is_match_success','match_time'
                        ])
                        match_results_df.set_index('index', inplace=True)
                        df = pd.concat([df, match_results_df], axis=1)

                    except Exception as e:
                        print(f"Error during matching for file {csv_path}: {e}")
                        all_reports.append({"file": csv_path, "error": f"Matching error: {e}"})
                        continue
                    
                    # Generate report
                    try:
                        match_success_count = int(df['is_match_success'].sum())
                        success_rate = float(df['is_match_success'].mean())
                        avg_match_time = float(df['match_time'].mean())

                        report = {
                            "file": csv_path,
                            "match_success_count": match_success_count,
                            "success_rate": success_rate,
                            "avg_match_time_seconds": avg_match_time
                        }
                        print(f"Report: {report}")

                        base_filename = os.path.splitext(os.path.basename(csv_path))[0]
                        save_json_report(save_root, task, dataset, subdir, base_filename, report)

                        # Add this:
                        try:
                            save_detailed_match_results(df, save_root, task, dataset, subdir, base_filename)
                        except Exception as e:
                            print(f"Error saving detailed match results for file {csv_path}: {e}")

                        all_reports.append(report)
                    except Exception as e:
                        print(f"Error during report generation for file {csv_path}: {e}")
                        all_reports.append({"file": csv_path, "error": f"Report generation error: {e}"})
                        continue

                    # Clear cache
                    if 'embedding_cache' in globals():
                        embedding_cache.clear()
                    if 'perturbed_embedding_cache' in globals():
                        perturbed_embedding_cache.clear()
                    del df
                    gc.collect()
                    torch.cuda.empty_cache()
    
    return all_reports

if __name__ == "__main__":
    set_random_seed(42)
    tokenizer = English()
    all_reports = process_csv_files()
    save_total_report(all_reports, 'Privacy_Results_FurtherAttackResults')