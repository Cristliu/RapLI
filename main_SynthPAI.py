import os
import sys
import json
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
from args import get_parser

from _01pii_detection.combined_detector import CombinedPIIDetector
from _02risk_assessment.risk_assessor import RiskAssessor
from _03dp_noise.noise_adder import DPNoiseAdder  # Noise adder
from data.data_loader_preprocess import DataLoader  # Import data loading and preprocessing class

# transformers related
from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

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

def load_model_and_tokenizer(model_name):
    """
    Load tokenizer and model based on the given model name, and move to GPU (if available)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer

def load_jsonl_data(file_path):
    """
    Load data in jsonl format, and extract text that needs to be processed
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if "comments" in item:
                    for comment in item["comments"]:
                        if "text" in comment:
                            # Save original data structure for later updates
                            data.append({
                                "username": item.get("username", ""),
                                "comment_text": comment["text"],
                                "comment_obj": comment,
                                "item_obj": item
                            })
            except json.JSONDecodeError:
                logging.warning(f"Unable to parse line: {line.strip()[:100]}...")
                continue
    return data

def save_disturbed_jsonl(data, output_path):
    """
    Save the perturbed text back to the original jsonl format
    """
    # First group by username to rebuild the original data structure
    grouped_data = {}
    for item in data:
        username = item["username"]
        if username not in grouped_data:
            grouped_data[username] = item["item_obj"]
            
        # Update the corresponding text in the comments section of the item
        for comment in grouped_data[username]["comments"]:
            if comment is item["comment_obj"]:
                comment["text"] = item["perturbed_sentence"]

    # Write to new jsonl file
    with open(output_path, 'w', encoding='utf-8') as f:
        for username, item_obj in grouped_data.items():
            f.write(json.dumps(item_obj, ensure_ascii=False) + '\n')


def calculate_privacy_budget(args, risk_level):
    return round(
        args.max_budget - (risk_level - 1) * ((args.max_budget - args.epsilon) / 4), 
        2
    )

def save_results(
    all_results, 
    args, 
    pii_detection_times, 
    risk_budget_times, 
    noise_adder_times, 
    results_dict,
    is_jsonl=False
):
    """
    Save perturbed results (all_results) to JSON and CSV, and record average times
    """
    
    out_dir = f"./NoiseResults/{args.task}/{args.dataset}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Construct file suffix based on ablation_1 ~ ablation_6
    ablation_list = []
    if args.ablation_1:
        ablation_list.append("ablation_1_")
    if args.ablation_2:
        ablation_list.append("ablation_2_")
    if args.ablation_3_1:
        ablation_list.append("ablation_3_1_")
    if args.ablation_3_2:
        ablation_list.append("ablation_3_2_")
    if args.ablation_3_3:
        ablation_list.append("ablation_3_3_")
    if args.ablation_4:
        ablation_list.append("ablation_4_")


    # If all are False, no suffix, otherwise join with "_"
    if len(ablation_list) == 0:
        ablation_suffix = ""
    else:
        #Set a separate folder for storing when ablation_list is not empty
        out_dir = f"./NoiseResults/{args.task}/{args.dataset}/Ours_Ablation"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        ablation_suffix = "_".join(ablation_list)

    # Add ablation_suffix to the filename
    json_path = os.path.join(
        out_dir,
        f"{ablation_suffix}Ours_epsilon_{args.epsilon}-{args.max_budget}.json"
    )
    csv_path = os.path.join(
        out_dir,
        f"{ablation_suffix}Ours_epsilon_{args.epsilon}-{args.max_budget}.csv"
    )
    time_json_path = os.path.join(
        out_dir,
        f"{ablation_suffix}Ours_epsilon_{args.epsilon}-{args.max_budget}_avg_pii_risk_addnoise_time.json"
    )
    epsilon_json_path = os.path.join(
        out_dir,
        f"{ablation_suffix}Ours_epsilon_{args.epsilon}-{args.max_budget}_avg_epsilon_S.json"
    )

    # If it's jsonl data, add an additional jsonl output file
    if is_jsonl:
        jsonl_path = os.path.join(
            out_dir,
            f"{ablation_suffix}Ours_epsilon_{args.epsilon}-{args.max_budget}_disturbed.jsonl"
        )
        save_disturbed_jsonl(all_results, jsonl_path)
        print(f"Perturbed JSONL file saved to: {jsonl_path}")
        
        # For compatibility with existing format, prepare new all_results for saving as JSON and CSV
        formatted_results = []
        for result in all_results:
            formatted_results.append({
                "text": result["comment_text"],
                "perturbed_sentence": result["perturbed_sentence"],
                "epsilon_S": result.get("epsilon_S", 0)
            })
        all_results = formatted_results

    # Save perturbation results (JSON)
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(all_results, json_file, ensure_ascii=False, indent=4)

    # Save perturbation results (CSV)
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

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

    # Also write to time_json file
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

    ###Ablation Experiment 1: Use bert-base-uncased directly for args.model_name
    if args.ablation_1:
        model, tokenizer = load_model_and_tokenizer("bert-base-uncased")
    else:
        model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Decide data loading method based on task type
    if args.task == "synthpai":
        # Process jsonl format data
        jsonl_path = f"./data/synthpai/comments_eval_revised.jsonl"  # Assuming data is in the data directory
        dataset = load_jsonl_data(jsonl_path)
        is_jsonl = True
    else:
        # Use existing data loading method
        data_loader = DataLoader(args.task, args.dataset)
        dataset = data_loader.load_and_preprocess()
        is_jsonl = False

    # Initialize PII detection, risk assessment, and noise addition classes
    detector = CombinedPIIDetector()
    risk_assessor = RiskAssessor()
    noise_adder = DPNoiseAdder(model, tokenizer, args)

    # Variables for the preparation process
    stop_words = set(stopwords.words('english'))
    special_tokens = set(tokenizer.all_special_tokens)
    punctuation_token = set(string.punctuation)
    keep_words = stop_words.union(special_tokens).union(punctuation_token)

    # Record time
    pii_detection_times = []
    risk_budget_times = []
    noise_adder_times = []

    all_results = []
    results_dict = {}

    for example in tqdm(dataset, desc=f"Processing {args.dataset}"):
        try:
            # Get text based on data type
            if is_jsonl:
                text = example["comment_text"]
                ground_truth = None  # jsonl data has no ground_truth
                hypothesis = None
            else:
                # Process existing data format
                if args.dataset == "ag_news" or args.dataset == "piidocs":
                    text = example["text"]
                    ground_truth = int(example["label"])
                    hypothesis = None

            # 1) PII detection
            st1 = time.time()
            combined_results = detector.detect_pii(text)
            pii_detection_times.append(round(time.time() - st1,2))

            # 2) Risk assessment & calculate privacy budget
            st2 = time.time()
            assessed_results = risk_assessor.assess_risk(combined_results)
            # Use task-specific tokenizer to encode text and get offset mapping
            encoded = tokenizer.encode_plus(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False
            )
            tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
            offsets = encoded['offset_mapping']

            # Create character-level risk level mapping
            char_risk_levels = {}
            for res in assessed_results:
                start = res['start']
                end = res['end']
                for i in range(start, end):
                    char_risk_levels[i] = res['risk_level']

            # Map risk levels to corresponding tokens
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
                    token_text = tokenizer.convert_tokens_to_string([token]).strip().lower()
                    if token_text in keep_words:
                        token_level = None
                    else:
                        token_level = random.randint(1, 3)  # Set default risk level
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
            
            ### Ablation Experiment 2: Don't calculate epsilon_S, directly use epsilon_i for epsilon_effective below
            if args.ablation_2:
                epsilon_S = 100  # Set a large value, so epsilon_effective directly uses epsilon_i below
            else:
                # Calculate total privacy budget epsilon_S for the sentence
                if len(sensitive_tokens) == 0:
                    epsilon_S = 100  # Not used
                else:
                    epsilon_S = sum([eps_i for eps_i in token_privacy_budgets if eps_i is not None]) / len(sensitive_tokens)

            risk_budget_times.append(round(time.time() - st2, 2))

            # 3). Generate candidate replacement words for sensitive tokens and randomly replace
            st3 = time.time()
                
            new_tokens = []
            for token, epsilon_i in zip(tokens, token_privacy_budgets):
                is_high_risk = token in high_risk_tokens

                if epsilon_i is not None:
                    epsilon_effective = min(epsilon_S, epsilon_i)
                    sim_words, probabilities = noise_adder.get_replacement(token, is_high_risk, epsilon_effective)
                    new_token = noise_adder.select_replacement(sim_words, probabilities)
                    new_tokens.append(new_token)
                else:
                    new_tokens.append(token)
            
            noise_adder_times.append(round(time.time() - st3, 2))

            # Combine into perturbed sentence
            perturbed_sentence = tokenizer.convert_tokens_to_string(new_tokens)

            # Collect results
            if is_jsonl:
                # Collect results for jsonl format data
                result = {
                    "username": example["username"],
                    "comment_text": text,
                    "perturbed_sentence": perturbed_sentence,
                    "epsilon_S": epsilon_S,
                    "comment_obj": example["comment_obj"],
                    "item_obj": example["item_obj"]
                }
            else:
                # Collect results for existing format data
                result = {
                    "text": text,
                    "ground_truth": ground_truth,
                    "perturbed_sentence": perturbed_sentence,
                    "epsilon_S": epsilon_S
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
                results_dict,
                is_jsonl=is_jsonl
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
        results_dict,
        is_jsonl=is_jsonl
    )


if __name__ == "__main__":
    main()