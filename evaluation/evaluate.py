# evaluate.py
import json
import os
import re
from sklearn.metrics import accuracy_score, f1_score
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
from bert_score import score as bertscore
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein
import numpy as np

# Try to load NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calc_bleu(reference, candidate):
    """Calculate BLEU score"""
    if not candidate or not reference:
        return 0
    
    smoothie = SmoothingFunction().method1
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    
    # Avoid empty token lists
    if not reference_tokens or not candidate_tokens:
        return 0
    
    # BLEU requires reference text list
    return sentence_bleu([reference_tokens], candidate_tokens, 
                         smoothing_function=smoothie)

def calc_levenshtein_similarity(reference, candidate):
    """Calculate Levenshtein similarity (1 - normalized_distance)"""
    if not candidate or not reference:
        return 0
    
    # Calculate Levenshtein distance
    distance = Levenshtein.distance(reference.lower(), candidate.lower())
    
    # Normalize distance (between 0 and 1)
    max_len = max(len(reference), len(candidate))
    if max_len == 0:
        return 1  # If both are empty, similarity is 1
    
    # Convert to similarity (1 - normalized distance)
    return 1 - (distance / max_len)

def calc_bertscore_batch(candidates, references, lang='en'):
    """Batch calculate BERTScore"""
    if not candidates or not references:
        return []
    
    try:
        # Calculate BERTScore (returns precision, recall, F1)
        # Select model based on language
        p, r, f1 = bertscore(candidates, references, lang=lang, verbose=False)
        # Convert to percentage and return F1 score
        scores = (100 * f1.detach().cpu().numpy()).tolist()
        return scores
    except Exception as e:
        print(f"BERTScore calculation error: {e}")
        return [0.0] * len(candidates)


def extract_summary_from_response(response, task="samsum"):
    """
    Extract summary information from LLM response - Hybrid method
    
    Try to extract using multiple methods by priority to ensure maximum compatibility:
    1. AST parsing (Most reliable)
    2. JSON parsing (Standard format)
    3. Improved Regex (Double quotes)
    4. Loose Regex (Single quotes)
    5. Fallback pattern matching
    
    Args:
        response: LLM response text
        task: Task type, used to adapt summary extraction for different languages
    """
    import ast
    import json
    
    # Method 1: AST parsing (Most reliable) - Priority use
    try:
        dict_match = re.search(r'\{[^}]*[\'"]summary[\'"][^}]*\}', response)
        if dict_match:
            dict_str = dict_match.group(0)
            try:
                parsed_dict = ast.literal_eval(dict_str)
                if isinstance(parsed_dict, dict) and 'summary' in parsed_dict:
                    return str(parsed_dict['summary']).strip()
            except (ValueError, SyntaxError):
                pass
    except Exception:
        pass
    
    # Method 2: JSON parsing (Handle standard JSON format)
    try:
        json_match = re.search(r'\{[^}]*"summary"[^}]*\}', response)
        if json_match:
            json_str = json_match.group(0)
            try:
                parsed_dict = json.loads(json_str)
                if isinstance(parsed_dict, dict) and 'summary' in parsed_dict:
                    return str(parsed_dict['summary']).strip()
            except (json.JSONDecodeError, ValueError):
                pass
    except Exception:
        pass
    
    # Method 3: Improved Regex (Handle content enclosed in double quotes)
    try:
        match = re.search(r'\{\s*[\'"]summary[\'"]\s*:\s*"([^"]*(?:""[^"]*)*)"\s*\}', response)
        if match:
            return match.group(1).strip()
    except Exception:
        pass
    
    # Method 4: Looser Regex (Single quote format, but smarter)
    try:
        pattern = r"""
            \{\s*                          # Start brace
            ['"]*summary['"]*\s*:\s*       # summary key
            '([^']*(?:                     # Start single quote, match content
                \\.[^']*                   # Content after escape character
            )*)'                           # End single quote
            \s*\}                          # End brace
        """
        match = re.search(pattern, response, re.VERBOSE)
        if match:
            result = match.group(1).replace("\\'", "'")
            return result.strip()
    except Exception:
        pass
    
    # Method 5: Final fallback
    try:
        patterns = [
            r"summary['\"]?\s*:\s*['\"]([^'\"]+)['\"]",
            r"summary['\"]?\s*:\s*['\"]([^'\"]*(?:[^'\"\\\\]|\\\\.)*)['\"]"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result = match.group(1).strip()
                if result:
                    return result
    except Exception:
        pass
    
    # If all methods fail, return empty string
    return ""

def evaluate_results(task, define_dataset, csv_file_name, ground_truths, predictions, black_llm_model_name, avg_time, responses=None):
    results = {}  
    
    if task in ["topic_classification", "sentiment_analysis", "clinical_inference", "piidocs_classification", "spam_email_classification"]:
        # Calculate basic metrics
        # Filter out failed predictions (label=100)
        valid_indices = [i for i, pred in enumerate(predictions) if pred != 100]
        
        if valid_indices:
            valid_ground_truths = [ground_truths[i] for i in valid_indices]
            valid_predictions = [predictions[i] for i in valid_indices]
            
            accuracy = accuracy_score(valid_ground_truths, valid_predictions)
            f1 = f1_score(valid_ground_truths, valid_predictions, average='weighted')
            
            results["accuracy"] = accuracy
            results["f1_score"] = f1
            results["avg_time"] = avg_time
            results["valid_samples"] = len(valid_indices)
            results["total_samples"] = len(predictions)
        else:
            results["accuracy"] = 0.0
            results["f1_score"] = 0.0
            results["avg_time"] = avg_time
            results["valid_samples"] = 0
            results["total_samples"] = len(predictions)
        
    elif task == "storycloze":
        # Evaluate StoryCloze task - Story Ending Prediction
        # Exclude failed predictions
        valid_indices = [i for i, pred in enumerate(predictions) if pred != 100]
        
        if valid_indices:
            valid_ground_truths = [ground_truths[i] for i in valid_indices]
            valid_predictions = [predictions[i] for i in valid_indices]
            
            # Calculate evaluation metrics
            accuracy = accuracy_score(valid_ground_truths, valid_predictions)
            f1 = f1_score(valid_ground_truths, valid_predictions, average='macro')
            
            results["accuracy"] = accuracy
            results["f1_score"] = f1
            results["avg_time"] = avg_time
        else:
            results["accuracy"] = 0.0
            results["f1_score"] = 0.0
            results["avg_time"] = avg_time
            
    # Modify summarization task part in evaluate_results function
    elif task in ["samsum", "lcsts"]:
        # Evaluate summarization task - Dialogue summarization or Chinese news summarization
        if responses is None:
            print(f"Error: {task} task requires responses parameter")
            results = {
                'rouge-1-recall': 0.0, 'rouge-1-precision': 0.0, 'rouge-1-f': 0.0,
                'rouge-2-recall': 0.0, 'rouge-2-precision': 0.0, 'rouge-2-f': 0.0,
                'rouge-l-recall': 0.0, 'rouge-l-precision': 0.0, 'rouge-l-f': 0.0,
                "bleu": 0.0, "levenshtein_similarity": 0.0, "bertscore": 0.0,
                "avg_time": avg_time
            }
        else:
            # Select stemmer based on task
            use_stemmer = True if task == "samsum" else False  # Chinese does not use stemmer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=use_stemmer)
            
            # Extract predicted summaries
            summaries = [extract_summary_from_response(response, task) for response in responses]
            
            # Calculate ROUGE metrics
            rouge_scores = {
                'rouge-1-recall': 0.0, 'rouge-1-precision': 0.0, 'rouge-1-f': 0.0,
                'rouge-2-recall': 0.0, 'rouge-2-precision': 0.0, 'rouge-2-f': 0.0,
                'rouge-l-recall': 0.0, 'rouge-l-precision': 0.0, 'rouge-l-f': 0.0
            }
            bleu_scores = []
            levenshtein_similarities = []
            
            # Collect valid samples for batch BERTScore calculation
            valid_summaries = []
            valid_references = []
            valid_count = 0
            
            for gt, summary in zip(ground_truths, summaries):
                # Check if summary is not empty
                if summary and gt:
                    try:
                        # Preprocess based on task type
                        if task == "samsum":
                            # English task uses sentence tokenization
                            ref_processed = "\n".join(sent_tokenize(gt))
                            summ_processed = "\n".join(sent_tokenize(summary))
                        else:
                            # Chinese task uses original text directly, no sentence tokenization
                            ref_processed = gt
                            summ_processed = summary
                        
                        # Calculate ROUGE scores for current sample
                        scores = scorer.score(ref_processed, summ_processed)
                        
                        # Save Recall, Precision and F-measure
                        rouge_scores['rouge-1-recall'] += scores['rouge1'].recall
                        rouge_scores['rouge-1-precision'] += scores['rouge1'].precision
                        rouge_scores['rouge-1-f'] += scores['rouge1'].fmeasure
                        
                        rouge_scores['rouge-2-recall'] += scores['rouge2'].recall
                        rouge_scores['rouge-2-precision'] += scores['rouge2'].precision
                        rouge_scores['rouge-2-f'] += scores['rouge2'].fmeasure
                        
                        rouge_scores['rouge-l-recall'] += scores['rougeLsum'].recall
                        rouge_scores['rouge-l-precision'] += scores['rougeLsum'].precision
                        rouge_scores['rouge-l-f'] += scores['rougeLsum'].fmeasure
                        
                        # BLEU score
                        bleu = calc_bleu(gt, summary)
                        bleu_scores.append(bleu)
                        
                        # Levenshtein similarity
                        lev_sim = calc_levenshtein_similarity(gt, summary)
                        levenshtein_similarities.append(lev_sim)
                        
                        # Collect valid samples for batch BERTScore calculation
                        valid_summaries.append(summary)
                        valid_references.append(gt)
                        
                        valid_count += 1
                    except Exception as e:
                        print(f"Error calculating metrics for: '{summary}' against '{gt}': {e}")
            
            # Batch calculate BERTScore
            print(f"Calculating BERTScore ({valid_count} valid samples)...")
            try:
                if valid_count > 0:
                    # Select language based on task
                    lang = 'zh' if task == "lcsts" else 'en'
                    bertscore_scores = calc_bertscore_batch(valid_summaries, valid_references, lang=lang)
                    avg_bertscore = np.mean(bertscore_scores) if bertscore_scores else 0.0
                else:
                    avg_bertscore = 0.0
            except Exception as e:
                print(f"BERTScore calculation failed: {e}")
                avg_bertscore = 0.0
            
            # Calculate averages
            results = {}
            print(f"Valid samples count: {valid_count}")
            if valid_count > 0:
                # Calculate average of all Rouge metrics and convert to percentage format (consistent with other scripts)
                for key in rouge_scores:
                    # ROUGE scores converted to percentage and rounded to 2 decimal places
                    results[key] = round((rouge_scores[key] / valid_count) * 100, 2)
                
                # Keep compatible with original format, while adding new detailed metrics
                results["rouge-1"] = results["rouge-1-f"]  # Keep original rouge-1 key for compatibility
                results["rouge-2"] = results["rouge-2-f"]  # Keep original rouge-2 key for compatibility
                results["rouge-l"] = results["rouge-l-f"]  # Keep original rouge-l key for compatibility
                
                results["bleu"] = round(np.mean(bleu_scores), 4) if bleu_scores else 0.0
                results["levenshtein_similarity"] = round(np.mean(levenshtein_similarities), 4) if levenshtein_similarities else 0.0
                results["bertscore"] = round(avg_bertscore, 2)  # Add BERTScore result
                results["avg_time"] = round(avg_time, 4)
            else:
                results = {
                    'rouge-1-recall': 0.0, 'rouge-1-precision': 0.0, 'rouge-1-f': 0.0,
                    'rouge-2-recall': 0.0, 'rouge-2-precision': 0.0, 'rouge-2-f': 0.0,
                    'rouge-l-recall': 0.0, 'rouge-l-precision': 0.0, 'rouge-l-f': 0.0,
                    'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0,
                    "bleu": 0.0, "levenshtein_similarity": 0.0, "bertscore": 0.0,
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