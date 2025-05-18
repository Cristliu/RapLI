import os
import glob
import time
import random
import json
import string
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM
from transformers import logging as transformers_logging
from nltk.corpus import stopwords


def set_random_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

def batch_inference(texts, tokenizer, model, device, top_k=5):
    """
    Batch mask inference attack
    """
    transformers_logging.set_verbosity_error()
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
    Mask inference attack
    """
    transformers_logging.set_verbosity_error()
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

    # Variables in preparation
    stop_words = set(stopwords.words('english'))
    punctuation_token = set(string.punctuation)
    stop_words = stop_words.union(punctuation_token)

    ROOT_DIR = "./NoiseResults/" 

    for task in ["piidocs_classification"]:
        if task == "topic_classification":
            dataset = "ag_news"
        elif task == "piidocs_classification":
            dataset = "piidocs"
        
        unsanitized_dir = os.path.join(ROOT_DIR, task, dataset, "Unsanitized")
        if not os.path.exists(unsanitized_dir):
            os.makedirs(unsanitized_dir, exist_ok=True)
        
        ours_csv_path = os.path.join(ROOT_DIR, task, dataset, f"Ours_epsilon_0.1-8.0.csv")
        unsanitized_csv_path = os.path.join(unsanitized_dir, "Unsanitized.csv")
        
        df = pd.read_csv(ours_csv_path)
        if task == "clinical_inference":
            df_unsanitized = df[["text", "ground_truth", "text", "hypothesis"]]
            df_unsanitized.columns = ["text", "ground_truth", "perturbed_sentence", "hypothesis"]
        else:
            df_unsanitized = df[["text", "ground_truth", "text"]]
            df_unsanitized.columns = ["text", "ground_truth", "perturbed_sentence"]
        
        df_unsanitized.to_csv(unsanitized_csv_path, index=False, encoding="utf-8-sig")
      
    
    ### Read the Unsanitized.csv files in the three Unsanitized folders above
    csv_file_list = glob.glob(os.path.join(ROOT_DIR, "**", "Unsanitized", "Unsanitized.csv"), recursive=True)
    
    # # Supplementary processing of piidocs_classification/piidocs/Unsanitized/Unsanitized.csv
    # csv_file_list = ["./NoiseResults/piidocs_classification/piidocs/Unsanitized/Unsanitized.csv"]

    
    print("=== Start traversing CSV files ===")
    print(f"Found {len(csv_file_list)} CSV files to process...")



    SAVE_DIR = "./NoiseResults_AttackResults"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)
    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.to(device)
    model.eval()

    # ============ Configuration section ============
    TOP_K = [1,5] # Loop through two K values
    BATCH_SIZE = 64
    DEBUG_MODE = False  # If you want to see detailed debug information, change to True


    for csv_file in csv_file_list:
        # Get the directory of the current csv_file
        csv_dir = os.path.dirname(csv_file)
        # Get the names of the two upper-level folders
        task = os.path.basename(os.path.dirname(os.path.dirname(csv_dir)))  # Get the parent directory's parent directory (e.g., topic_classification)
        dataset = os.path.basename(os.path.dirname(csv_dir))  # Get the parent directory (e.g., ag_news)

        # Construct save path
        save_subfolder = os.path.join(SAVE_DIR, task, dataset, "Unsanitized")
        os.makedirs(save_subfolder, exist_ok=True)
        
        for K_item in TOP_K:

            file_name = os.path.splitext(os.path.basename(csv_file))[0]

            print(f"\nProcessing CSV file: {csv_file}")

            # 1. Read data
            df = pd.read_csv(csv_file)
            if not all(col in df.columns for col in ["text", "perturbed_sentence"]):
                print(f"File {csv_file} does not contain required columns: ['text', 'perturbed_sentence'], skipping.")
                continue

            # ============ 1. Record the time of the current file ============
            # Initialize target columns as object type (if not already initialized)
            if "mask_success_words" not in df.columns:
                df["mask_success_words"] = [[] for _ in range(len(df))]
            if "mask_Expstop_success_words" not in df.columns:
                df["mask_Expstop_success_words"] = [[] for _ in range(len(df))]
                
            # print(f"mask_success_words content: {df['mask_success_words']}")    
            # print(f"mask_Expstop_success_words content: {df['mask_Expstop_success_words']}")
        
            file_start_time = time.time()

            # Used to count the processing time of each record in this file
            row_times = []

 
            # 2. Iterate through each record
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
                original_text = str(row["text"])
                perturbed_text = str(row["perturbed_sentence"])

                row_start_time = time.time()

                mask_success_rate, Mask_success_words, mask_rate_ExpStop, Mask_Expstop_success_words = mask_token_inference_attack_subword_topk_batch(
                    [], [],
                    tokenizer.tokenize(original_text),
                    tokenizer.tokenize(perturbed_text),
                    tokenizer,
                    model,
                    device,
                    top_k=K_item,
                    batch_size=BATCH_SIZE,
                    stop_words=stop_words,
                    debug=DEBUG_MODE
                )

                row_end_time = time.time()
                row_time = row_end_time - row_start_time
                row_times.append(row_time)

                # Record mask_success_rate, Mask_success_words, mask_rate_ExpStop, Mask_Expstop_success_words and write back to df
                # print(f"Type of Mask_success_words: {type(Mask_success_words)}")
                # print(f"Content of Mask_success_words: {Mask_success_words}")
                df.at[idx, f"mask_rate"] = mask_success_rate
                df.at[idx, f"mask_rate_ExpStop"] = mask_rate_ExpStop
                df.at[idx, f"mask_success_words"] = Mask_success_words
                df.at[idx, f"mask_Expstop_success_words"] = Mask_Expstop_success_words
                # Print the current row of df
                # print(f"Current row of df: {df.iloc[idx]}")
                # Debug - only process the first two records
                # if idx >= 1:
                #     return 0

            # 3. Count time
            file_end_time = time.time()
            file_time = file_end_time - file_start_time

            avg_row_time = sum(row_times) / len(row_times) if len(row_times) > 0 else 0.0

            # 4. Count average mask_success_rate mask_rate_ExpStop
            mean_r_ats = df[f"mask_rate"].mean()
            mean_mask_rate_ExpStop = df[f"mask_rate_ExpStop"].mean()

            print(f"\nStatistics for file {csv_file}:")
            print(f"  - Average attack success rate (Top-{K_item}): {mean_r_ats:.4f}")
            print(f"  - Average attack success rate (ExpStop Top-{K_item}): {mean_mask_rate_ExpStop:.4f}")
            print(f"  - Total processing time for this file: {file_time:.4f} seconds")
            print(f"  - Average processing time per record: {avg_row_time:.4f} seconds")

            
            # Save results to NoiseResults_AttackResults\topic_classification\ag_news\Unsanitized; NoiseResults_AttackResults\sentiment_analysis\Unsanitized; NoiseResults_AttackResults\clinical_inference\mednli\Unsanitized folders as csv and json files
            

            # === Save csv file ===
            updated_csv_path = os.path.join(save_subfolder, f"{file_name}_MaskInfer_{K_item}.csv")
            df.to_csv(updated_csv_path, index=False, encoding="utf-8-sig")
            print(f"Updated CSV file saved to: {updated_csv_path}")

            # === Save JSON evaluation results ===
            result_json = {
                "mean_mask_success_rate": mean_r_ats,
                "mean_mask_rate_ExpStop": mean_mask_rate_ExpStop,
                "file_processing_time_s": round(file_time, 4),
                "per_record_avg_time_s": round(avg_row_time, 4)
            }
            json_file_path = os.path.join(save_subfolder, f"{file_name}_MaskInfer_{K_item}_attackrate_time.json")
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=4)
            print(f"Evaluation results JSON saved to: {json_file_path}")


if __name__ == "__main__":
    main()
