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
from data.data_loader_preprocess import DataLoader
from _01pii_detection.combined_detector import CombinedPIIDetector
from _02risk_assessment.risk_assessor import RiskAssessor
from _03dp_noise.noise_adder_Glove_Attacks import DPNoiseAdder  
from spacy.lang.en import English
from transformers import BertTokenizer, BertForMaskedLM
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

def save_results(all_results, args, pii_detection_times, risk_budget_times, noise_adder_times, results_dict):
    """
    Save the perturbed results (all_results) to JSON and CSV, and record the average time
    """
    out_dir = f"./NoiseResults_AttackResults/{args.task}/{args.dataset}/Ours_Ablation"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"Ours_GloVe_epsilon_{args.epsilon}-{args.max_budget}_EmbInver_{args.EmbInver_K}_MaskInfer_{args.MaskInfer_K}.json")
    csv_path = os.path.join(out_dir, f"Ours_GloVe_epsilon_{args.epsilon}-{args.max_budget}_EmbInver_{args.EmbInver_K}_MaskInfer_{args.MaskInfer_K}.csv")
    time_json_path = os.path.join(out_dir, f"Ours_GloVe_epsilon_{args.epsilon}-{args.max_budget}_EmbInver_{args.EmbInver_K}_MaskInfer_{args.MaskInfer_K}_avg_pii_risk_addnoise_attack_time.json")
    epsilon_json_path = os.path.join(out_dir, f"Ours_GloVe_epsilon_{args.epsilon}-{args.max_budget}_EmbInver_{args.EmbInver_K}_MaskInfer_{args.MaskInfer_K}_avg_epsilon_S.json")
    avg_attack_results_path = os.path.join(out_dir, f"Ours_GloVe_epsilon_{args.epsilon}-{args.max_budget}_EmbInver_{args.EmbInver_K}_MaskInfer_{args.MaskInfer_K}_avg_attack_results.json")

    # Save perturbed results (JSON)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(all_results, jf, ensure_ascii=False, indent=4)

    # Save perturbed results (CSV)
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Record and save the time for each stage
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

    # Calculate and save the average value of epsilon_S
    epsilon_values = [item["epsilon_S"] for item in all_results if "epsilon_S" in item]
    avg_epsilon_S = float(sum(epsilon_values) / len(epsilon_values)) if epsilon_values else 0.0
    with open(epsilon_json_path, "w", encoding="utf-8") as ef:
        json.dump({"avg_epsilon_S": avg_epsilon_S}, ef, ensure_ascii=False, indent=4)

    # Calculate and save the average attack success rate
    avg_embInver_rate = np.mean([item["embInver_rate"] for item in all_results])
    avg_embExpStop_rate = np.mean([item["embExpStop_rate"] for item in all_results])
    avg_maskInfer_rate = np.mean([item["mask_rate"] for item in all_results])
    avg_mask_rate_ExpStop = np.mean([item["mask_rate_ExpStop"] for item in all_results])

    avg_attack_results = {
        "avg_embInver_rate": avg_embInver_rate,
        "avg_embExpStop_rate": avg_embExpStop_rate,
        "avg_maskInfer_rate": avg_maskInfer_rate,
        "avg_mask_rate_ExpStop": avg_mask_rate_ExpStop,
    }
    with open(avg_attack_results_path, "w", encoding="utf-8") as arf:
        json.dump(avg_attack_results, arf, ensure_ascii=False, indent=4)

def batch_inference(texts, tokenizer, model, device, top_k=5):
    """
    Batch inference for masked token attacks
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

    batch_predictions = []
    for b_idx in range(len(texts)):
        mask_token_index = torch.where(inputs["input_ids"][b_idx] == tokenizer.mask_token_id)[0]
        if len(mask_token_index) == 0:
            batch_predictions.append([])
            continue
        mask_idx = mask_token_index[0].item()
        mask_logits = logits[b_idx, mask_idx, :]
        topk_ids = torch.topk(mask_logits, k=top_k, dim=-1).indices.tolist()
        topk_subtokens = [tokenizer.convert_ids_to_tokens(x) for x in topk_ids]
        batch_predictions.append(topk_subtokens)
    return batch_predictions

def mask_token_inference_attack_subword_topk_batch(
        Mask_success_words,
        Mask_Expstop_success_words,
        original_subtokens,
        perturbed_subtokens,
        tokenizer,
        model,
        device,
        top_k=5,
        batch_size=32,
        stop_words=None,
        debug=False
    ):
    """
    Mask token inference attack
    """
    if len(original_subtokens) != len(perturbed_subtokens):
        print(f"\n Warning: Subword length mismatch: {len(original_subtokens)} vs {len(perturbed_subtokens)}")
    min_len = min(len(original_subtokens), len(perturbed_subtokens))
    total_subtokens = min_len

    batch_texts = []
    batch_indices = []
    all_topk_predictions = {}

    for i in range(min_len):
        temp_subtokens = perturbed_subtokens.copy()
        temp_subtokens[i] = tokenizer.mask_token
        masked_input_str = " ".join(temp_subtokens)
        batch_texts.append(masked_input_str)
        batch_indices.append(i)

        if len(batch_texts) == batch_size:
            batch_results = batch_inference(batch_texts, tokenizer, model, device, top_k=top_k)
            for b_idx, preds in enumerate(batch_results):
                subword_pos = batch_indices[b_idx]
                all_topk_predictions[subword_pos] = preds
            batch_texts = []
            batch_indices = []

    if len(batch_texts) > 0:
        batch_results = batch_inference(batch_texts, tokenizer, model, device, top_k=top_k)
        for b_idx, preds in enumerate(batch_results):
            subword_pos = batch_indices[b_idx]
            all_topk_predictions[subword_pos] = preds

    matched_count = 0
    matched_ExpStop_count = 0
    for i in range(min_len):
        topk_subtokens = all_topk_predictions.get(i, [])
        if original_subtokens[i].lower() in topk_subtokens:
            matched_count += 1
            Mask_success_words.append(original_subtokens[i])
            if original_subtokens[i].lower() not in stop_words:
                matched_ExpStop_count += 1
                Mask_Expstop_success_words.append(original_subtokens[i])

    r_ats = matched_count / total_subtokens if total_subtokens > 0 else 0.0
    r_ats_ExpStop = matched_ExpStop_count / total_subtokens if total_subtokens > 0 else 0.0
    return r_ats, Mask_success_words, r_ats_ExpStop, Mask_Expstop_success_words

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

    # Initialize
    data_loader = DataLoader(args.task, args.dataset)
    dataset = data_loader.load_and_preprocess()
    detector = CombinedPIIDetector()
    risk_assessor = RiskAssessor()
    tokenizer = English()
    noise_adder = DPNoiseAdder(args, "./data/embeddings/glove_840B-300d.txt")

    stop_words = set(stopwords.words('english'))
    # special_tokens = set(tokenizer.all_special_tokens)
    punctuation_token = set(string.punctuation)
    keep_words = stop_words.union(punctuation_token)

    # Record time
    pii_detection_times = []
    risk_budget_times = []
    noise_adder_times = []

    all_results = []
    results_dict = {}

    # Load BERT model and tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    bert_model.to(device)
    bert_model.eval()

    for example in tqdm(dataset, desc=f"Processing {args.dataset}"):
        try:
            # ag_news: text, label
            # SemEval: text, label
            # mednli: text, label, hypothesis
            if args.dataset == "ag_news" or args.dataset == "piidocs":
                text = example["text"]
                ground_truth = int(example["label"])
                hypothesis = None

            # 1) PII detection
            st1 = time.time()
            combined_results = detector.detect_pii(text)
            pii_detection_times.append(round(time.time() - st1,2))


            doc = tokenizer(text)
            tokens = [token.text for token in doc]
            offsets = [(token.idx, token.idx + len(token.text)) for token in doc]

            # 2) Risk assessment & calculate privacy budget
            st2 = time.time()
            assessed_results = risk_assessor.assess_risk(combined_results)

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
                    token_text = token.strip().lower()
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
            
            # Calculate the total privacy budget epsilon_S for the sentence
            if len(sensitive_tokens) == 0:
                epsilon_S = 100  # Not used
            else:
                epsilon_S = sum([eps_i for eps_i in token_privacy_budgets if eps_i is not None]) / len(sensitive_tokens)

            risk_budget_times.append(round(time.time() - st2, 2))

            # 3) Generate candidate replacement words for sensitive tokens and perform random replacement
            st3 = time.time()
                
            new_tokens = []
            EmbInver_success_cnt = 0
            EmbExpStop_success_cnt = 0
            EmbInver_success_words = []
            EmbExpStop_success_words = []
            Mask_success_words = []
            Mask_Expstop_success_words = []

            for token, epsilon_i in zip(tokens, token_privacy_budgets):
                is_high_risk = token in high_risk_tokens

                if epsilon_i is not None:
                    epsilon_effective = min(epsilon_S, epsilon_i)
                    sim_words, probabilities = noise_adder.get_glove_replacement(token, is_high_risk, epsilon_effective)
                    new_token = noise_adder.select_replacement(token, sim_words, probabilities)
                    new_tokens.append(new_token)

                    # Embedding Inversion Attack
                    similar_words_K = noise_adder.EmbInver_attack(new_token, args.EmbInver_K)
                    if token .lower() in [w.lower() for w in similar_words_K]:
                        EmbInver_success_cnt += 1
                        EmbInver_success_words.append(token)
                        if token.lower() not in keep_words:
                            EmbExpStop_success_cnt += 1
                            EmbExpStop_success_words.append(token)
                else:
                    new_tokens.append(token)
                    EmbInver_success_cnt += 1
                    EmbInver_success_words.append(token)
                    if token.lower() not in keep_words:
                        EmbExpStop_success_cnt += 1
                        EmbExpStop_success_words.append(token)
            
            noise_adder_times.append(round(time.time() - st3,2))

            # Calculate EmbInver success rate, the total number of attacks is the length of tokens
            embInver_success_rate = EmbInver_success_cnt / len(tokens) if len(tokens) > 0 else 0.0
            embExpStop_rate = EmbExpStop_success_cnt / len(tokens) if len(tokens) > 0 else 0.0

            # Mask Token Inference Attack
            mask_success_rate, Mask_success_words, mask_rate_ExpStop, Mask_Expstop_success_words = mask_token_inference_attack_subword_topk_batch(
                Mask_success_words, Mask_Expstop_success_words, tokens, new_tokens, bert_tokenizer, bert_model,
                device, top_k=args.MaskInfer_K, batch_size=64, stop_words=keep_words
            )

            perturbed_sentence = " ".join(new_tokens)
            
            result = {
                "text": text,
                "ground_truth": ground_truth,
                "perturbed_sentence": perturbed_sentence,
                "epsilon_S": epsilon_S,
                "embInver_rate": embInver_success_rate,
                "embInver_success_words": EmbInver_success_words,
                "embExpStop_rate": embExpStop_rate,
                "embExpStop_success_words": EmbExpStop_success_words,
                "mask_rate": mask_success_rate,
                "mask_success_words": Mask_success_words,
                "mask_rate_ExpStop": mask_rate_ExpStop,
                "mask_Expstop_success_words": Mask_Expstop_success_words,
            }
            if args.dataset == "mednli" and hypothesis is not None:
                result["hypothesis"] = hypothesis
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