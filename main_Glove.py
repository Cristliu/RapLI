import os
import sys
import json
import csv
import time
import string
import random
import logging
import traceback

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords

# Similar to main.py, but using GloVe Embedding, without considering ablation parameters
from data.data_loader_preprocess import DataLoader
from _01pii_detection.combined_detector import CombinedPIIDetector
from _02risk_assessment.risk_assessor import RiskAssessor
from _03dp_noise.noise_adder_Glove import DPNoiseAdder  # Using GloVe-based noise adder
from spacy.lang.en import English
from args import get_parser



device = "cuda" if torch.cuda.is_available() else "cpu"

def set_random_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

def calculate_privacy_budget(args, risk_level):
    return round(
        args.max_budget - (risk_level - 1) * ((args.max_budget - args.epsilon) / 4), 
        2
    )

def save_results(all_results, args, pii_detection_times, risk_budget_times, noise_adder_times,results_dict):
    """
    Save perturbed results (all_results) to JSON and CSV, and record average times
    """
    out_dir = f"./NoiseResults/{args.task}/{args.dataset}/Ours_Ablation"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"Ours_GloVe_epsilon_{args.epsilon}-{args.max_budget}.json")
    csv_path = os.path.join(out_dir, f"Ours_GloVe_epsilon_{args.epsilon}-{args.max_budget}.csv")
    time_json_path = os.path.join(out_dir, f"Ours_GloVe_epsilon_{args.epsilon}-{args.max_budget}_avg_pii_risk_addnoise_time.json")
    epsilon_json_path = os.path.join(out_dir, f"Ours_GloVe_epsilon_{args.epsilon}-{args.max_budget}_avg_epsilon_S.json")

    # Save perturbation results (JSON)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(all_results, jf, ensure_ascii=False, indent=4)

    # Save perturbation results (CSV)
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Record and save time for each stage
    results_dict["pii_detection_avg_time"] = (
        sum(pii_detection_times) / len(pii_detection_times) if pii_detection_times else 0
    )
    results_dict["risk_budget_avg_time"] = (
        sum(risk_budget_times) / len(risk_budget_times) if risk_budget_times else 0
    )
    results_dict["noise_adder_avg_time"] = (
        sum(noise_adder_times) / len(noise_adder_times) if noise_adder_times else 0
    )
    with open(time_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)

    # Calculate and save average epsilon_S
    epsilon_values = [item["epsilon_S"] for item in all_results if "epsilon_S" in item]
    avg_epsilon_S = float(sum(epsilon_values) / len(epsilon_values)) if epsilon_values else 0.0
    with open(epsilon_json_path, "w", encoding="utf-8") as ef:
        json.dump({"avg_epsilon_S": avg_epsilon_S}, ef, ensure_ascii=False, indent=4)

def main():
    set_random_seed(42)
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Running => task: {args.task}, dataset: {args.dataset}, epsilon: {args.epsilon}, max_budget: {args.max_budget}")
    logger.info(f"Model Name: {args.model_name}")

    # Initialize DataLoader
    data_loader = DataLoader(args.task, args.dataset)
    dataset = data_loader.load_and_preprocess()

    # Initialize components
    detector = CombinedPIIDetector()
    risk_assessor = RiskAssessor()
    tokenizer = English()
    noise_adder = DPNoiseAdder(args, "./data/embeddings/glove_840B-300d.txt")

    # Auxiliary variables
    stop_words = set(stopwords.words('english'))
    punctuation_token = set(string.punctuation)
    keep_words = stop_words.union(punctuation_token)

    # Time statistics
    pii_detection_times = []
    risk_budget_times = []
    noise_adder_times = []

    all_results = []
    results_dict = {}

    for example in tqdm(dataset, desc=f"Processing {args.dataset} with GloVe"):
        try:
            if args.dataset == "ag_news" or args.dataset == "piidocs":
                text = example["text"]
                ground_truth = int(example["label"])
                hypothesis = None
            
            elif args.dataset == "samsum":
                text = example["text"]
                ground_truth = example["label"]  # This is the summary text
                hypothesis = None
            

            st1 = time.time()
            combined_results = detector.detect_pii(text)
            pii_detection_times.append(round(time.time() - st1,2))


            doc = tokenizer(text)
            tokens = [token.text for token in doc]
            offsets = [(token.idx, token.idx + len(token.text)) for token in doc]

            st2 = time.time()
            assessed_results = risk_assessor.assess_risk(combined_results)
        
            # Create character-level risk mapping
            char_risk_levels = {}
            for res in assessed_results:
                start = res['start']
                end = res['end']
                for i in range(start, end):
                    char_risk_levels[i] = res['risk_level']

            # # Map risk levels to corresponding tokens ### This step generally maps the highest risk token
            token_risk_levels = []
            token_privacy_budgets = []
            high_risk_tokens = []
            sensitive_tokens = []
            non_sensitive_tokens = []

            for token, (start_offset, end_offset) in zip(tokens, offsets):
                token_start = start_offset
                token_end = end_offset
                token_text = text[token_start:token_end]
                token_level = None
                for i in range(token_start, token_end):
                    if i in char_risk_levels:
                        if token_level is None or char_risk_levels[i] > token_level:
                            token_level = char_risk_levels[i]
                if token_level is None:
                    token_text = token.strip().lower()
                    if token_text in keep_words:
                        token_level = None
                    else:
                        token_level = random.randint(1, 3)  # Set default risk level
                        # print(f"\nToken: {token} -> {token_level}")
                token_risk_levels.append(token_level)
                
                if token_level is not None:
                    epsilon_i = calculate_privacy_budget(args, token_level)
                    sensitive_tokens.append((token, epsilon_i))
                    if token_level == 5:
                        high_risk_tokens.append(token)
                else:
                    epsilon_i = None
                    non_sensitive_tokens.append(token)
                
                token_privacy_budgets.append(epsilon_i)

            if len(sensitive_tokens) == 0:
                epsilon_S = 100 # Not used
            else:
                epsilon_S = sum([eps_i for eps_i in token_privacy_budgets if eps_i is not None]) / len(sensitive_tokens)

            risk_budget_times.append(round(time.time() - st2,2))

            st3 = time.time()
            new_tokens = []
            for token, epsilon_i in zip(tokens, token_privacy_budgets):
                # If token is in high_risk_tokens, assign True, otherwise False
                is_high_risk = token in high_risk_tokens

                if epsilon_i is not None:
                    epsilon_effective = min(epsilon_S, epsilon_i)
                    sim_words, probabilities = noise_adder.get_glove_replacement(token, is_high_risk, epsilon_effective)
                    new_token = noise_adder.select_replacement(token,sim_words, probabilities)
                    new_tokens.append(new_token)
                else:
                    new_tokens.append(token)
            noise_adder_times.append(round(time.time() - st3,2))

            perturbed_sentence = " ".join(new_tokens)

            result = {
                "text": text,
                "ground_truth": ground_truth,
                "perturbed_sentence": perturbed_sentence,
                "epsilon_S": epsilon_S,
            }
            

            all_results.append(result)

        except KeyboardInterrupt:
            print("Terminating early. Saving current results...")
            save_results(
                all_results,
                args,
                pii_detection_times,
                risk_budget_times,
                noise_adder_times,
                results_dict
            )
            sys.exit(1)
            return
        
        except Exception as e:
            logger.error(f"Error while processing example: {e}")
            logger.error(traceback.format_exc())
            continue
    
    # After processing, save results
    save_results(
        all_results,
        args,
        pii_detection_times,
        risk_budget_times,
        noise_adder_times,
        results_dict
    )
if __name__ == "__main__":
    main()