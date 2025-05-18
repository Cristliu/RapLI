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
import re

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
    results_dict
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
    
    # Add l1 or l4 suffix (if provided)
    level_suffix = ""
    if hasattr(args, 'suffix'):
        level_suffix = f"_{args.suffix}"

    # Add ablation_suffix and level_suffix to the filename
    json_path = os.path.join(
        out_dir,
        f"{ablation_suffix}Ours_highmask_epsilon_{args.epsilon}-{args.max_budget}{level_suffix}.json"
    )
    csv_path = os.path.join(
        out_dir,
        f"{ablation_suffix}Ours_highmask_epsilon_{args.epsilon}-{args.max_budget}{level_suffix}.csv"
    )
    time_json_path = os.path.join(
        out_dir,
        f"{ablation_suffix}Ours_highmask_epsilon_{args.epsilon}-{args.max_budget}{level_suffix}_avg_pii_risk_addnoise_time.json"
    )
    epsilon_json_path = os.path.join(
        out_dir,
        f"{ablation_suffix}Ours_highmask_epsilon_{args.epsilon}-{args.max_budget}{level_suffix}_avg_epsilon_S.json"
    )

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

def mask_text_with_entities(text, combined_results):
    """
    Split text from left to right based on entity (start, end) indices, then insert [MASK] at positions that need replacement.
    Also record the replaced content.
    """
    # Sort by start index in ascending order
    entity_spans = sorted((res["start"], res["end"]) for res in combined_results)

    # List for concatenation
    output_chunks = []
    prev_end = 0
    replaced_entities = []

    for (start_idx, end_idx) in entity_spans:
        # First add the unreplaced portion [prev_end, start_idx)
        output_chunks.append(text[prev_end:start_idx])
        # Record the replaced content
        replaced_entities.append(text[start_idx:end_idx])
        # Insert [MASK] in the replacement segment ===> Using MASK conflicts with later processing ===> Using [] other content would be split by tokenizer, so use a word that won't be split
        output_chunks.append("[UNK]")
        prev_end = end_idx

    # Add the last segment
    output_chunks.append(text[prev_end:])

    # Concatenate and clean up (merge extra spaces, etc.)
    masked_text = "".join(output_chunks)
    masked_text = re.sub(r'\s+', ' ', masked_text).strip()
    return masked_text, replaced_entities

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

    data_loader = DataLoader(args.task, args.dataset)
    dataset = data_loader.load_and_preprocess()
    print(dataset[0].keys())

    # Initialize PII detection, risk assessment, and noise addition classes
    detector = CombinedPIIDetector()
    risk_assessor = RiskAssessor()
    noise_adder = DPNoiseAdder(model, tokenizer, args)

    # Variables for the preparation process
    stop_words = set(stopwords.words('english'))
    special_tokens = set(tokenizer.all_special_tokens)
    punctuation_token = set(string.punctuation)
    keep_words = stop_words.union(special_tokens).union(punctuation_token)

    # Prepare separate result collection lists and timers for three experiment groups
    # Original method
    all_results_original = []
    pii_detection_times_original = []
    risk_budget_times_original = []
    noise_adder_times_original = []
    results_dict_original = {}
    
    # L1 control group (token_level=1)
    all_results_l1 = []
    pii_detection_times_l1 = []
    risk_budget_times_l1 = []
    noise_adder_times_l1 = []
    results_dict_l1 = {}
    
    # L4 control group (token_level=4)
    all_results_l4 = []
    pii_detection_times_l4 = []
    risk_budget_times_l4 = []
    noise_adder_times_l4 = []
    results_dict_l4 = {}

    for example in tqdm(dataset, desc=f"Processing {args.dataset}"):
        try:
            # ag_news: text, label
            # SemEval: text, label
            # mednli: text, label, hypothesis
            # samsum: text, label (summary)
            # storycloze: text, ending1, ending2, correct_ending
            if args.dataset == "ag_news" or args.dataset == "piidocs":
                text = example["text"]
                ground_truth = int(example["label"])
                hypothesis = None
            elif args.dataset == "samsum":
                text = example["text"]
                ground_truth = example["label"]  # This is the summary text
                hypothesis = None


            # 1) PII detection - shared by all experiment groups
            st1 = time.time()
            combined_results = detector.detect_pii(text)
            pii_detection_time = round(time.time() - st1, 2)
            pii_detection_times_original.append(pii_detection_time)
            pii_detection_times_l1.append(pii_detection_time)
            pii_detection_times_l4.append(pii_detection_time)

            # 2) Risk assessment & masking process - shared by all experiment groups
            st2 = time.time()
            masked_text, replaced_entities = mask_text_with_entities(text, combined_results)
            
            # Check mask consistency
            mask_cnt = masked_text.count("[UNK]")
            replaced_entities_cnt = len(replaced_entities)
            if mask_cnt != replaced_entities_cnt:
                print(f"Warning: [UNK] count {mask_cnt} != replaced_entities count {replaced_entities_cnt}")

            # Encode the text
            encoded = tokenizer.encode_plus(
                masked_text,
                return_offsets_mapping=True,
                add_special_tokens=False
            )
            tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
            offsets = encoded['offset_mapping']

            # Create character-level risk level mapping
            char_risk_levels = {}
            risk_assessment_time = round(time.time() - st2, 2)
            
            #######################################################
            # Original method processing - randomly assign risk levels
            #######################################################
            st2_original = time.time()
            
            # Process token risk levels and privacy budgets for original method
            token_risk_levels_original = []
            token_privacy_budgets_original = []
            high_risk_tokens_original = []
            sensitive_tokens_original = []
            non_sensitive_tokens_original = []

            for token, (start_offset, end_offset) in zip(tokens, offsets):
                token_start = start_offset
                token_end = end_offset
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
                        token_level = random.randint(1, 4)  # Original method: random risk level
                token_risk_levels_original.append(token_level)
                
                if token_level is not None:
                    epsilon_i = calculate_privacy_budget(args, token_level)
                    sensitive_tokens_original.append((token, epsilon_i))
                    if token_level == 5:
                        high_risk_tokens_original.append(token)
                else:
                    epsilon_i = None
                    non_sensitive_tokens_original.append(token)
                
                token_privacy_budgets_original.append(epsilon_i)
            
            # Calculate total privacy budget epsilon_S for the sentence
            if args.ablation_2:
                epsilon_S_original = 100  # Ablation experiment 2: use larger value directly
            else:
                if len(sensitive_tokens_original) == 0:
                    epsilon_S_original = 100
                else:
                    epsilon_S_original = sum([eps_i for eps_i in token_privacy_budgets_original if eps_i is not None]) / len(sensitive_tokens_original)

            risk_budget_times_original.append(round(time.time() - st2_original, 2))
            
            #######################################################
            # L1 control group processing - all token_level set to 1
            #######################################################
            st2_l1 = time.time()
            
            # Process token risk levels and privacy budgets for L1 control group
            token_risk_levels_l1 = []
            token_privacy_budgets_l1 = []
            high_risk_tokens_l1 = []
            sensitive_tokens_l1 = []
            non_sensitive_tokens_l1 = []

            for token, (start_offset, end_offset) in zip(tokens, offsets):
                token_start = start_offset
                token_end = end_offset
                token_text = tokenizer.convert_tokens_to_string([token]).strip().lower()
                
                if token_text in keep_words:
                    token_level = None
                else:
                    token_level = 1  # L1 control group: all non-keep_words tokens set to level 1
                
                token_risk_levels_l1.append(token_level)
                
                if token_level is not None:
                    epsilon_i = calculate_privacy_budget(args, token_level)
                    sensitive_tokens_l1.append((token, epsilon_i))
                    if token_level == 5:
                        high_risk_tokens_l1.append(token)
                else:
                    epsilon_i = None
                    non_sensitive_tokens_l1.append(token)
                
                token_privacy_budgets_l1.append(epsilon_i)
            
            # Calculate total privacy budget epsilon_S for the sentence
            if args.ablation_2:
                epsilon_S_l1 = 100
            else:
                if len(sensitive_tokens_l1) == 0:
                    epsilon_S_l1 = 100
                else:
                    epsilon_S_l1 = sum([eps_i for eps_i in token_privacy_budgets_l1 if eps_i is not None]) / len(sensitive_tokens_l1)

            risk_budget_times_l1.append(round(time.time() - st2_l1, 2))
            
            #######################################################
            # L4 control group processing - all token_level set to 4
            #######################################################
            st2_l4 = time.time()
            
            # Process token risk levels and privacy budgets for L4 control group
            token_risk_levels_l4 = []
            token_privacy_budgets_l4 = []
            high_risk_tokens_l4 = []
            sensitive_tokens_l4 = []
            non_sensitive_tokens_l4 = []

            for token, (start_offset, end_offset) in zip(tokens, offsets):
                token_start = start_offset
                token_end = end_offset
                token_text = tokenizer.convert_tokens_to_string([token]).strip().lower()
                
                if token_text in keep_words:
                    token_level = None
                else:
                    token_level = 4  # L4 control group: all non-keep_words tokens set to level 4
                
                token_risk_levels_l4.append(token_level)
                
                if token_level is not None:
                    epsilon_i = calculate_privacy_budget(args, token_level)
                    sensitive_tokens_l4.append((token, epsilon_i))
                    if token_level == 5:
                        high_risk_tokens_l4.append(token)
                else:
                    epsilon_i = None
                    non_sensitive_tokens_l4.append(token)
                
                token_privacy_budgets_l4.append(epsilon_i)
            
            # Calculate total privacy budget epsilon_S for the sentence
            if args.ablation_2:
                epsilon_S_l4 = 100
            else:
                if len(sensitive_tokens_l4) == 0:
                    epsilon_S_l4 = 100
                else:
                    epsilon_S_l4 = sum([eps_i for eps_i in token_privacy_budgets_l4 if eps_i is not None]) / len(sensitive_tokens_l4)

            risk_budget_times_l4.append(round(time.time() - st2_l4, 2))
            
            #######################################################
            # 3) Add noise processing - separate for three experiment groups
            #######################################################
            
            # Original method adds noise
            st3_original = time.time()
            new_tokens_original = []
            for token, epsilon_i in zip(tokens, token_privacy_budgets_original):
                is_high_risk = token in high_risk_tokens_original
                if epsilon_i is not None:
                    epsilon_effective = min(epsilon_S_original, epsilon_i)
                    sim_words, probabilities = noise_adder.get_replacement(token, is_high_risk, epsilon_effective)
                    new_token = noise_adder.select_replacement(sim_words, probabilities)
                    new_tokens_original.append(new_token)
                else:
                    new_tokens_original.append(token)
            noise_adder_times_original.append(round(time.time() - st3_original, 2))
            perturbed_sentence_original = tokenizer.convert_tokens_to_string(new_tokens_original)
            
            # L1 control group adds noise
            st3_l1 = time.time()
            new_tokens_l1 = []
            for token, epsilon_i in zip(tokens, token_privacy_budgets_l1):
                is_high_risk = token in high_risk_tokens_l1
                if epsilon_i is not None:
                    epsilon_effective = min(epsilon_S_l1, epsilon_i)
                    sim_words, probabilities = noise_adder.get_replacement(token, is_high_risk, epsilon_effective)
                    new_token = noise_adder.select_replacement(sim_words, probabilities)
                    new_tokens_l1.append(new_token)
                else:
                    new_tokens_l1.append(token)
            noise_adder_times_l1.append(round(time.time() - st3_l1, 2))
            perturbed_sentence_l1 = tokenizer.convert_tokens_to_string(new_tokens_l1)
            
            # L4 control group adds noise
            st3_l4 = time.time()
            new_tokens_l4 = []
            for token, epsilon_i in zip(tokens, token_privacy_budgets_l4):
                is_high_risk = token in high_risk_tokens_l4
                if epsilon_i is not None:
                    epsilon_effective = min(epsilon_S_l4, epsilon_i)
                    sim_words, probabilities = noise_adder.get_replacement(token, is_high_risk, epsilon_effective)
                    new_token = noise_adder.select_replacement(sim_words, probabilities)
                    new_tokens_l4.append(new_token)
                else:
                    new_tokens_l4.append(token)
            noise_adder_times_l4.append(round(time.time() - st3_l4, 2))
            perturbed_sentence_l4 = tokenizer.convert_tokens_to_string(new_tokens_l4)
            
            # Collect results - separate for three experiment groups
            # Create basic result dictionary for each experiment group
            result_original = {
                "text": text,
                "ground_truth": ground_truth,
                "perturbed_sentence": perturbed_sentence_original,
                "epsilon_S": epsilon_S_original
            }
            
            result_l1 = {
                "text": text,
                "ground_truth": ground_truth,
                "perturbed_sentence": perturbed_sentence_l1,
                "epsilon_S": epsilon_S_l1
            }
            
            result_l4 = {
                "text": text,
                "ground_truth": ground_truth,
                "perturbed_sentence": perturbed_sentence_l4,
                "epsilon_S": epsilon_S_l4
            }
            

            
            # Add results to the corresponding result lists
            all_results_original.append(result_original)
            all_results_l1.append(result_l1)
            all_results_l4.append(result_l4)

        except KeyboardInterrupt:
            print("Terminating early. Saving current results...")
            # Save results for three experiment groups
            save_results(all_results_original, args, pii_detection_times_original, 
                        risk_budget_times_original, noise_adder_times_original, results_dict_original)
            
            # Add suffix for L1 and L4 control groups
            import copy
            args_l1 = copy.deepcopy(args)
            args_l1.suffix = "l1"
            save_results(all_results_l1, args_l1, pii_detection_times_l1, 
                        risk_budget_times_l1, noise_adder_times_l1, results_dict_l1)
            
            args_l4 = copy.deepcopy(args)
            args_l4.suffix = "l4"
            save_results(all_results_l4, args_l4, pii_detection_times_l4, 
                        risk_budget_times_l4, noise_adder_times_l4, results_dict_l4)
            
            sys.exit(1)
            return
        
        except Exception as e:
            logger.error(f"Error while processing example: {e}")
            logger.error(traceback.format_exc())
            continue
        
    # After processing, save results for three experiment groups
    save_results(all_results_original, args, pii_detection_times_original, 
                risk_budget_times_original, noise_adder_times_original, results_dict_original)
    
    # Add suffix for L1 and L4 control groups
    import copy
    args_l1 = copy.deepcopy(args)
    args_l1.suffix = "l1"
    save_results(all_results_l1, args_l1, pii_detection_times_l1, 
                risk_budget_times_l1, noise_adder_times_l1, results_dict_l1)
    
    args_l4 = copy.deepcopy(args)
    args_l4.suffix = "l4"
    save_results(all_results_l4, args_l4, pii_detection_times_l4, 
                risk_budget_times_l4, noise_adder_times_l4, results_dict_l4)




if __name__ == "__main__":
    main()