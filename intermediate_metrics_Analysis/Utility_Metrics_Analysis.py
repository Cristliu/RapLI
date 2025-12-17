# Utility_Metrics_Analysis.py
"""
Calculate utility metrics for perturbed text:
1. Perplexity (PPL) - Perplexity of the perturbed text
2. BLEU - BLEU score between perturbed text and original text
3. ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) - ROUGE scores between perturbed text and original text
4. Token Error Rate (TER) - Token-level error rate
"""

import json
import os
import sys
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import gc
import random
import torch
import time
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Import Chinese segmentation tool
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("Warning: jieba not available. Install with: pip install jieba")

# Download NLTK resources
def ensure_nltk_resources():
    """Ensure NLTK resources are available"""
    resources_to_download = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab')
    ]
    
    for resource_path, resource_name in resources_to_download:
        try:
            nltk.data.find(resource_path)
            print(f"OK: NLTK resource '{resource_name}' already available")
        except LookupError:
            print(f"Downloading NLTK resource '{resource_name}'...")
            try:
                nltk.download(resource_name, quiet=True)
                print(f"OK: Successfully downloaded '{resource_name}'")
            except Exception as e:
                print(f"WARNING: Failed to download '{resource_name}': {e}")
                print(f"  Will use fallback tokenization for this resource")

# Ensure NLTK resources
ensure_nltk_resources()

# Set random seed
def set_random_seed(seed_value=42):
    """Fix random seed to ensure reproducibility"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

class PerplexityCalculator:
    """
    Calculate perplexity using GPT-2
    Supports two modes:
    1. standard: Standard mode, directly calculate perplexity of perturbed text
    2. teacher_forcing: Calculate in the context of the original document
    
    Supports Chinese and English datasets:
    - English: Use 'gpt2' model
    - Chinese: Use 'uer/gpt2-chinese-cluecorpussmall' model
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', mode='standard', is_chinese=False):
        self.device = device
        self.mode = mode
        self.is_chinese = is_chinese
        
        # Select model based on language
        if is_chinese:
            model_name = 'uer/gpt2-chinese-cluecorpussmall'
            print(f"Loading Chinese GPT-2 model '{model_name}' for perplexity calculation on {device} (mode: {mode})...")
        else:
            model_name = 'gpt2'
            print(f"Loading English GPT-2 model '{model_name}' for perplexity calculation on {device} (mode: {mode})...")
        
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        
        # Chinese model uses BertTokenizer, needs to be loaded with AutoTokenizer
        if is_chinese:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        
        self.model.eval()
        print(f"GPT-2 model loaded successfully! (Language: {'Chinese' if is_chinese else 'English'})")
    
    def calculate_perplexity(self, text):
        """Calculate perplexity for a single text (standard mode)"""
        if not text or len(text.strip()) == 0:
            return float('inf')
        
        try:
            encodings = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
            input_ids = encodings.input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return perplexity
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return float('inf')
    
    def calculate_perplexity_teacher_forcing(self, original_text, perturbed_text):
        """
        Calculate perplexity using Teacher Forcing: Calculate probability of perturbed text given original document D context
        
        Implementation:
        1. Concatenate original text and perturbed text: [original_text] + [perturbed_text]
        2. Calculate loss only for the perturbed text part (mask out the original text part)
        3. PPL(perturbed | original) = exp(-1/N * Σ log P(x_i | original, x_<i))
        
        This evaluates: The "reasonableness" of the perturbed text given the original document context
        
        Args:
            original_text: Original text (as context condition)
            perturbed_text: Perturbed text (to be evaluated)
        
        Returns:
            perplexity: Conditional perplexity value
        """
        if not perturbed_text or len(perturbed_text.strip()) == 0:
            return float('inf')
        
        if not original_text or len(original_text.strip()) == 0:
            # If no original text, fallback to standard mode
            return self.calculate_perplexity(perturbed_text)
        
        try:
            # Method 1: Concatenate context (original text + perturbed text)
            # Use special separator to mark two parts
            combined_text = f"{original_text} [SEP] {perturbed_text}"
            
            # Tokenize original text and perturbed text
            original_encodings = self.tokenizer(
                original_text, 
                return_tensors='pt', 
                truncation=True, 
                max_length=512
            )
            original_length = original_encodings.input_ids.shape[1]
            
            # Tokenize complete combined text
            combined_encodings = self.tokenizer(
                combined_text,
                return_tensors='pt',
                truncation=True,
                max_length=1024
            )
            input_ids = combined_encodings.input_ids.to(self.device)
            
            # Create labels, calculate loss only for perturbed text part
            # Set label for original text part to -100 (PyTorch will ignore this part)
            labels = input_ids.clone()
            labels[0, :original_length + 2] = -100  # +2 because of [SEP] token
            
            # Calculate conditional perplexity
            with torch.no_grad():
                outputs = self.model(input_ids, labels=labels)
                loss = outputs.loss
                
                # If loss is nan (possibly because all masked), use standard method
                if torch.isnan(loss):
                    return self.calculate_perplexity(perturbed_text)
                
                perplexity = torch.exp(loss).item()
            
            return perplexity
            
        except Exception as e:
            print(f"Error in teacher forcing perplexity, falling back to standard: {e}")
            # Fallback to standard mode on error
            return self.calculate_perplexity(perturbed_text)

class BLEUCalculator:
    """Calculate BLEU score"""
    def __init__(self):
        self.smoothing = SmoothingFunction().method1
    
    def calculate_bleu(self, reference, candidate, is_chinese=False):
        """
        Calculate BLEU score (0-100)
        Args:
            reference: Reference text
            candidate: Candidate text
            is_chinese: Whether it is Chinese text
        """
        if not candidate or not reference:
            return 0.0
        
        try:
            # Select appropriate tokenization method based on language
            if is_chinese:
                if JIEBA_AVAILABLE:
                    # Use jieba segmentation
                    reference_tokens = list(jieba.cut(reference.lower()))
                    candidate_tokens = list(jieba.cut(candidate.lower()))
                else:
                    # Fallback to character level
                    reference_tokens = list(reference.lower())
                    candidate_tokens = list(candidate.lower())
            else:
                # English: Use NLTK tokenizer or fallback
                try:
                    reference_tokens = nltk.word_tokenize(reference.lower())
                    candidate_tokens = nltk.word_tokenize(candidate.lower())
                except LookupError:
                    # Fallback: simple split if NLTK tokenizer fails
                    print("Warning: NLTK tokenizer failed, using simple split")
                    reference_tokens = reference.lower().split()
                    candidate_tokens = candidate.lower().split()
            
            if not reference_tokens or not candidate_tokens:
                return 0.0
            
            # Calculate BLEU (0-1 range)
            bleu_score = sentence_bleu(
                [reference_tokens], 
                candidate_tokens,
                smoothing_function=self.smoothing
            )
            
            # Convert to 0-100 range
            return bleu_score * 100
        except Exception as e:
            print(f"Error calculating BLEU: {e}")
            return 0.0

class ROUGECalculator:
    """Calculate ROUGE score"""
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
    
    def calculate_rouge(self, reference, candidate):
        """
        Calculate ROUGE score
        Returns: dict with rouge1, rouge2, rougeL (F-scores, 0-100 range)
        """
        if not candidate or not reference:
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0
            }
        
        try:
            scores = self.scorer.score(reference, candidate)
            
            return {
                'rouge1': scores['rouge1'].fmeasure * 100,
                'rouge2': scores['rouge2'].fmeasure * 100,
                'rougeL': scores['rougeL'].fmeasure * 100
            }
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0
            }

class TERCalculator:
    """Calculate Token Error Rate (similar to edit distance)"""
    def __init__(self):
        pass
    
    def calculate_ter(self, reference, candidate, is_chinese=False, use_char_level=False):
        """
        Calculate Token Error Rate
        TER = (Insertions + Deletions + Substitutions) / Reference text token count
        Returns: Error rate in 0-100 range
        Args:
            reference: Reference text
            candidate: Candidate text
            is_chinese: Whether it is Chinese text
            use_char_level: Whether to force character-level segmentation (only valid for Chinese)
        """
        if not reference:
            return 100.0 if candidate else 0.0
        
        try:
            # Select appropriate tokenization method based on language
            if is_chinese:
                if use_char_level:
                    # Force character level
                    ref_tokens = list(reference.lower().strip())
                    cand_tokens = list(candidate.lower().strip())
                elif JIEBA_AVAILABLE:
                    # Use jieba segmentation
                    ref_tokens = list(jieba.cut(reference.lower()))
                    cand_tokens = list(jieba.cut(candidate.lower()))
                else:
                    # Fallback to character level
                    ref_tokens = list(reference.lower())
                    cand_tokens = list(candidate.lower())
            else:
                # English: Use NLTK tokenizer or fallback
                try:
                    ref_tokens = nltk.word_tokenize(reference.lower())
                    cand_tokens = nltk.word_tokenize(candidate.lower())
                except LookupError:
                    # Fallback: simple split if NLTK tokenizer fails
                    print("Warning: NLTK tokenizer failed for TER, using simple split")
                    ref_tokens = reference.lower().split()
                    cand_tokens = candidate.lower().split()
            
            if not ref_tokens:
                return 100.0 if cand_tokens else 0.0
            
            # Calculate edit distance (Levenshtein distance)
            edit_distance = self._levenshtein_distance(ref_tokens, cand_tokens)
            
            # TER = Edit distance / Reference length
            ter = (edit_distance / len(ref_tokens)) * 100
            
            return min(ter, 100.0)  # Limit to 100
        except Exception as e:
            print(f"Error calculating TER: {e}")
            return 100.0
    
    def _levenshtein_distance(self, ref_tokens, cand_tokens):
        """Calculate Levenshtein distance between two token sequences"""
        m, n = len(ref_tokens), len(cand_tokens)
        
        # Create DP matrix
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == cand_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # Deletion
                        dp[i][j-1],    # Insertion
                        dp[i-1][j-1]   # Substitution
                    )
        
        return dp[m][n]

def process_csv_files(ppl_mode='both'):
    """
    Process all CSV files and calculate utility metrics
    
    Args:
        ppl_mode: Perplexity calculation mode
                 - 'standard': Only calculate standard mode
                 - 'teacher_forcing': Only calculate Teacher Forcing mode
                 - 'both': Calculate both modes (default)
    """
    
    # Determine modes to compute
    modes_to_compute = []
    if ppl_mode == 'both':
        modes_to_compute = ['standard', 'teacher_forcing']
    else:
        modes_to_compute = [ppl_mode]
    
    print(f"Will compute Perplexity in mode(s): {', '.join(modes_to_compute)}")
    
    # Initialize non-PPL calculators (these are language-independent)
    print("Initializing calculators...")
    bleu_calc = BLEUCalculator()
    rouge_calc = ROUGECalculator()
    ter_calc = TERCalculator()
    print("Non-perplexity calculators initialized!\n")
    
    # Define tasks and corresponding datasets
    tasks_datasets = {
        # "topic_classification": ["ag_news"],
        "piidocs_classification": ["piidocs"],
        # "samsum": ["samsum"],
        # "spam_email_classification": ["spam_email"],
    }
    
    # Define all possible subdirectories
    subdirectories = [""]  # Can add more subdirectories, e.g., ["", "Ours_Ablation"]
    
    # Define root directory for saving results
    save_root = 'Utility_Analysis_Results'
    
    # Cache for PPL calculators of different languages
    ppl_calculators = {}  # key: 'chinese' or 'english', value: PerplexityCalculator instance
    
    # Iterate through each task and dataset
    for task, datasets in tasks_datasets.items():
        for dataset in datasets:
            for subdir in subdirectories:
                # Build CSV file read path
                if subdir:
                    csv_pattern = os.path.join('NoiseResults', task, dataset, subdir, '*.csv')
                    # csv_paths = glob.glob(csv_pattern)
                    # csv_paths = [csv_path for csv_path in csv_paths if "Kan_" in csv_path]
                else:
                    csv_pattern = os.path.join('NoiseResults',  task, dataset, '*.csv')
                    # csv_paths = glob.glob(csv_pattern)
                    # csv_paths = [csv_path for csv_path in csv_paths if "Kan_" in csv_path]
                
                csv_paths = glob.glob(csv_pattern)
                
                if not csv_paths:
                    print(f"Warning: No CSV files found in path {csv_pattern}, skipping.")
                    continue
                
                for csv_path in tqdm(csv_paths, desc=f"Processing {task}/{dataset}/{subdir}"):
                    # Read CSV file
                    try:
                        df = pd.read_csv(csv_path)
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")
                        continue
                    
                    # Determine which column to use for perturbed sentence
                    perturbed_col = 'perturbed_sentence'
                    if 'samsum_improved' in csv_path.replace('\\', '/'):
                        perturbed_col = 'polished_sentence'
                        print(f"Using '{perturbed_col}' as perturbed source for {csv_path}")

                    # Check necessary columns
                    if 'text' not in df.columns or perturbed_col not in df.columns:
                        print(f"Warning: {csv_path} missing necessary columns ('text' or '{perturbed_col}'), skipping.")
                        continue
                    
                    # Determine if it is a Chinese dataset
                    is_chinese = dataset in ['lcsts', 'spam_email']  # Adjust based on actual situation
                    
                    # Prepare PPL calculator
                    # For Chinese datasets, we need to calculate PPL for both Chinese model (uer) and English model (gpt2)
                    active_ppl_calcs = [] # list of (label, calculator_instance)
                    
                    if is_chinese:
                        # 1. Chinese model
                        if 'chinese' not in ppl_calculators:
                            print(f"\nInitializing CHINESE Perplexity Calculator...")
                            ppl_calculators['chinese'] = PerplexityCalculator(is_chinese=True)
                        active_ppl_calcs.append(('zh', ppl_calculators['chinese']))
                        
                        # 2. English model (for comparison)
                        if 'english' not in ppl_calculators:
                            print(f"\nInitializing ENGLISH Perplexity Calculator...")
                            ppl_calculators['english'] = PerplexityCalculator(is_chinese=False)
                        active_ppl_calcs.append(('en', ppl_calculators['english']))
                    else:
                        # English datasets only use English model
                        if 'english' not in ppl_calculators:
                            print(f"\nInitializing ENGLISH Perplexity Calculator...")
                            ppl_calculators['english'] = PerplexityCalculator(is_chinese=False)
                        active_ppl_calcs.append(('en', ppl_calculators['english']))

                    # Initialize result storage
                    results_data = {
                        'bleu': [],
                        'rouge1': [],
                        'rouge2': [],
                        'rougeL': [],
                        'ter_default': [], # Jieba for Chinese, Word for English
                        'ter_char': [],    # Char for Chinese
                        'ppl': {},         # key: f"{label}_{mode}" -> list
                        'times': []
                    }
                    
                    # Initialize PPL list
                    for label, _ in active_ppl_calcs:
                        for mode in modes_to_compute:
                            results_data['ppl'][f"{label}_{mode}"] = []
                    
                    # Iterate through each record
                    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Calculating metrics", leave=False):
                        original_text = str(row.get('text', ''))
                        perturbed_text = str(row.get(perturbed_col, ''))
                        
                        start_time = time.time()
                        
                        # 1. Calculate Perplexity (supports multiple models)
                        for label, calc in active_ppl_calcs:
                            if 'standard' in modes_to_compute:
                                ppl = calc.calculate_perplexity(perturbed_text)
                                results_data['ppl'][f"{label}_standard"].append(ppl)
                            
                            if 'teacher_forcing' in modes_to_compute:
                                ppl = calc.calculate_perplexity_teacher_forcing(original_text, perturbed_text)
                                results_data['ppl'][f"{label}_teacher_forcing"].append(ppl)
                        
                        # 2. Calculate BLEU
                        bleu = bleu_calc.calculate_bleu(original_text, perturbed_text, is_chinese=is_chinese)
                        results_data['bleu'].append(bleu)
                        
                        # 3. Calculate ROUGE
                        rouge_result = rouge_calc.calculate_rouge(original_text, perturbed_text)
                        results_data['rouge1'].append(rouge_result['rouge1'])
                        results_data['rouge2'].append(rouge_result['rouge2'])
                        results_data['rougeL'].append(rouge_result['rougeL'])
                        
                        # 4. Calculate TER
                        # Default method (Jieba for Chinese, Word for English)
                        ter_def = ter_calc.calculate_ter(original_text, perturbed_text, is_chinese=is_chinese, use_char_level=False)
                        results_data['ter_default'].append(ter_def)
                        
                        # If Chinese, additionally calculate character-level TER
                        if is_chinese:
                            ter_char = ter_calc.calculate_ter(original_text, perturbed_text, is_chinese=True, use_char_level=True)
                            results_data['ter_char'].append(ter_char)
                        
                        end_time = time.time()
                        results_data['times'].append(end_time - start_time)
                    
                    # Calculate average metrics
                    result_json = {
                        "total_samples": len(df)
                    }
                    
                    # Calculate PPL average
                    for key, values in results_data['ppl'].items():
                        valid_ppls = [p for p in values if p != float('inf')]
                        avg_ppl = float(np.mean(valid_ppls)) if valid_ppls else float('inf')
                        result_json[f'avg_perplexity_{key}'] = avg_ppl
                        result_json[f'valid_samples_{key}'] = len(valid_ppls)

                    # Calculate other metrics
                    result_json['avg_bleu'] = float(np.mean(results_data['bleu'])) if results_data['bleu'] else 0.0
                    result_json['avg_rouge1'] = float(np.mean(results_data['rouge1'])) if results_data['rouge1'] else 0.0
                    result_json['avg_rouge2'] = float(np.mean(results_data['rouge2'])) if results_data['rouge2'] else 0.0
                    result_json['avg_rougeL'] = float(np.mean(results_data['rougeL'])) if results_data['rougeL'] else 0.0
                    
                    result_json['avg_ter'] = float(np.mean(results_data['ter_default'])) if results_data['ter_default'] else 0.0
                    if is_chinese and results_data['ter_char']:
                        result_json['avg_ter_char'] = float(np.mean(results_data['ter_char']))
                        
                    result_json['avg_processing_time_seconds'] = float(np.mean(results_data['times'])) if results_data['times'] else 0.0
                    
                    # Print statistics
                    print(f"\n{csv_path} Statistics:")
                    for key in sorted(results_data['ppl'].keys()):
                        print(f"  - Perplexity ({key}): {result_json[f'avg_perplexity_{key}']:.4f}")
                        
                    print(f"  - Avg BLEU: {result_json['avg_bleu']:.4f}")
                    print(f"  - Avg ROUGE-1: {result_json['avg_rouge1']:.4f}")
                    print(f"  - Avg ROUGE-2: {result_json['avg_rouge2']:.4f}")
                    print(f"  - Avg ROUGE-L: {result_json['avg_rougeL']:.4f}")
                    print(f"  - Avg TER (Default): {result_json['avg_ter']:.4f}%")
                    if is_chinese:
                        print(f"  - Avg TER (Char-level): {result_json.get('avg_ter_char', 0.0):.4f}%")
                    print(f"  - Avg Processing Time: {result_json['avg_processing_time_seconds']:.4f} seconds")
                    
                    # Build save path
                    json_save_dir = os.path.join(save_root, task, dataset, subdir)
                    os.makedirs(json_save_dir, exist_ok=True)
                    
                    # Build JSON filename (excluding mode info)
                    base_filename = os.path.splitext(os.path.basename(csv_path))[0]
                    json_filename = f"Utility_Metrics_{base_filename}.json"
                    json_path = os.path.join(json_save_dir, json_filename)
                    
                    # Save JSON file
                    try:
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(result_json, f, ensure_ascii=False, indent=4)
                        print(f"  → Saved to: {json_path}")
                    except Exception as e:
                        print(f"Error saving JSON {json_path}: {e}")
                    
                    # Save detailed results
                    try:
                        save_detailed_results_combined(
                            df,
                            results_data,
                            save_root,
                            task,
                            dataset,
                            subdir,
                            base_filename,
                            perturbed_col=perturbed_col
                        )
                    except Exception as e:
                        print(f"Error saving detailed results: {e}")
                    
                    # Release memory
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

def save_detailed_results_combined(
    df, 
    results_data,
    save_root, 
    task, 
    dataset, 
    subdir, 
    base_filename,
    perturbed_col='perturbed_sentence'
):
    """Save detailed metrics for each record"""
    detailed_save_dir = os.path.join(save_root, task, dataset, subdir, "detailed_metrics")
    os.makedirs(detailed_save_dir, exist_ok=True)
    
    # Prepare detailed results
    detailed_results = []
    for i, row in df.iterrows():
        result = {
            "text": row["text"],
            "perturbed_sentence": row[perturbed_col]
        }
        
        # Add all PPL
        for key, values in results_data['ppl'].items():
            ppl_value = values[i]
            result[f"perplexity_{key}"] = float(ppl_value) if ppl_value != float('inf') else "inf"
        
        # Add other metrics
        result.update({
            "bleu": float(results_data['bleu'][i]),
            "rouge1": float(results_data['rouge1'][i]),
            "rouge2": float(results_data['rouge2'][i]),
            "rougeL": float(results_data['rougeL'][i]),
            "ter_default": float(results_data['ter_default'][i])
        })
        
        # If character-level TER exists
        if results_data['ter_char']:
            result["ter_char"] = float(results_data['ter_char'][i])
            
        detailed_results.append(result)
    
    # Save to JSON file
    detailed_json_path = os.path.join(detailed_save_dir, f"Detailed_Utility_Metrics_{base_filename}.json")
    try:
        with open(detailed_json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=4)
        print(f"  → Detailed results: {detailed_json_path}")
    except Exception as e:
        print(f"Error saving detailed metrics to {detailed_json_path}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Utility Metrics Analysis')
    parser.add_argument('--ppl_mode', type=str, default='both', 
                        choices=['standard', 'teacher_forcing', 'both'],
                        help='Perplexity calculation mode: standard, teacher_forcing, or both (default: both)')
    args = parser.parse_args()
    
    print("="*80)
    print("Utility Metrics Analysis")
    print("Computing: Perplexity, BLEU, ROUGE, Token Error Rate (TER)")
    print(f"Perplexity Mode: {args.ppl_mode}")
    if args.ppl_mode == 'both':
        print("  → Will compute BOTH standard and teacher_forcing modes")
    print("="*80)
    
    # Ensure NLTK resources are available before processing
    print("Checking NLTK resources...")
    ensure_nltk_resources()
    
    set_random_seed(42)
    process_csv_files(ppl_mode=args.ppl_mode)
    
    print("\n" + "="*80)
    print("Analysis completed!")
    print("="*80)
