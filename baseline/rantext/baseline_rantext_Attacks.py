import csv
import json
import os
import signal
import time
import numpy as np
import tqdm 
from args_rantext import *
import func_rantext_Attacks as func
import torch
import random
import logging
from transformers import BertTokenizer, BertForMaskedLM

# ==================== Load token embeddings ====================
def set_random_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

def read_data(file_path, dataset):
    data = []
    # Display loading progress
    with open(file_path, 'r', encoding='utf-8') as file:
        header = next(file)
        for line in tqdm.tqdm(file, desc=f"Loading {file_path}"):
            if dataset == "ag_news":
                label, sentence = line.strip().split('\t')
                data.append((sentence, label.strip()))
            elif dataset == "SemEval":
                _, label, sentence = line.strip().split('\t')               
                data.append((sentence, label.strip()))
            elif dataset == "mednli":
                label, sentence, hypothesis = line.strip().split('\t')
                data.append((sentence, label.strip(), hypothesis.strip()))  # hypothesis.strip() removes spaces from both sides
            elif dataset == "piidocs":
                label, sentence = line.strip().split('\t')
                data.append((sentence, label.strip()))
    return data

def save_results(perturbed_data, json_file, csv_file, start_time, dataset):
    if dataset == "ag_news" or dataset == "SemEval" or dataset == "piidocs" or dataset == "piidocs":
        json_data = [
            {
                "text": sentence,
                "ground_truth": label,
                "perturbed_sentence": perturbed_tokens,
                "embInver_rate": EmbInver_success_rate,
                "embExpStop_rate": EmbExpStop_success_rate,
                "embInver_success_words": EmbInver_success_words,
                "embExpStop_success_words": EmbExpStop_success_words,
                "mask_rate": mask_rate,
                "mask_rate_ExpStop": mask_rate_ExpStop,
                "mask_success_words": Mask_success_words,
                "mask_Expstop_success_words": Mask_Expstop_success_words
            }
            for sentence, label, perturbed_tokens, EmbInver_success_rate, EmbExpStop_success_rate,EmbInver_success_words,EmbExpStop_success_words, mask_rate, mask_rate_ExpStop, Mask_success_words,  Mask_Expstop_success_words in perturbed_data
        ]
    elif dataset == "mednli":
        json_data = [
            {
                "text": sentence,
                "ground_truth": label,
                "perturbed_sentence": perturbed_tokens,
                "hypothesis": hypothesis,
                "embInver_rate": EmbInver_success_rate,
                "embExpStop_rate": EmbExpStop_success_rate,
                "embInver_success_words": EmbInver_success_words,
                "embExpStop_success_words": EmbExpStop_success_words,
                "mask_rate": mask_rate,
                "mask_rate_ExpStop": mask_rate_ExpStop,
                "mask_success_words": Mask_success_words,
                "mask_Expstop_success_words": Mask_Expstop_success_words
            }
            for sentence, label, hypothesis, perturbed_tokens, EmbInver_success_rate, EmbExpStop_success_rate,EmbInver_success_words,EmbExpStop_success_words, mask_rate, mask_rate_ExpStop, Mask_success_words,Mask_Expstop_success_words in perturbed_data
        ]

    with open(json_file, 'w', encoding='utf-8') as jf:
        json.dump(json_data, jf, ensure_ascii=False, indent=4)

    with open(csv_file, 'w', encoding='utf-8', newline='') as cf:
        writer = csv.writer(cf)
        if dataset == "ag_news" or dataset == "SemEval" or dataset == "piidocs":
            writer.writerow(['text', 'ground_truth', 'perturbed_sentence', 'embInver_rate', 'embExpStop_rate', 'embInver_success_words', 'embExpStop_success_words', 'mask_rate', 'mask_rate_ExpStop', 'mask_success_words', 'mask_Expstop_success_words'])
        elif dataset == "mednli":
            writer.writerow(['text', 'ground_truth', 'hypothesis', 'perturbed_sentence', 'embInver_rate', 'embExpStop_rate', 'embInver_success_words', 'embExpStop_success_words', 'mask_rate', 'mask_rate_ExpStop', 'mask_success_words', 'mask_Expstop_success_words'])
        for row in perturbed_data:
            writer.writerow(row)

    end_time = time.time()
    runtime = end_time - start_time
    avg_runtime_data = {
        "avg_runtime_seconds": runtime / len(perturbed_data) if len(perturbed_data) > 0 else 0,
    }
    runtime_file = json_file.replace('.json', '_avg_addnoise_time.json')
    with open(runtime_file, 'w', encoding='utf-8') as rf:
        json.dump(avg_runtime_data, rf, ensure_ascii=False, indent=4)

    # Save avg_success_rate to JSON
    if dataset == "ag_news" or dataset == "SemEval" or dataset == "piidocs":
        avg_success_rate_data = {
            "avg_embInver_rate": np.mean([EmbInver_success_rate for _, _, _, EmbInver_success_rate, _, _, _, _, _, _, _ in perturbed_data]),
            "avg_embExpStop_rate": np.mean([EmbExpStop_success_rate for _, _, _, _, EmbExpStop_success_rate, _, _, _, _, _, _ in perturbed_data]),
            "avg_mask_rate": np.mean([mask_rate for _, _, _, _, _, _, _, mask_rate, _,_,_ in perturbed_data]),
            "avg_mask_rate_ExpStop": np.mean([mask_rate_ExpStop for _, _, _, _, _, _, _, _, mask_rate_ExpStop, _, _ in perturbed_data]),
        }
    elif dataset == "mednli":
        avg_success_rate_data = {
            "avg_embInver_rate": np.mean([EmbInver_success_rate for _, _, _, _, EmbInver_success_rate, _, _, _, _, _, _,_ in perturbed_data]),
            "avg_embExpStop_rate": np.mean([EmbExpStop_success_rate for _, _, _, _, _, EmbExpStop_success_rate, _, _, _, _, _,_ in perturbed_data]),
            "avg_mask_rate": np.mean([mask_rate for _, _, _, _, _, _, _, _, mask_rate, _,_,_ in perturbed_data]),
            "avg_mask_rate_ExpStop": np.mean([mask_rate_ExpStop for _, _, _, _, _, _, _, _,_, mask_rate_ExpStop, _, _ in perturbed_data]),
        }
    
    success_rate_file = json_file.replace('.json', '_avg_success_rate.json')
    with open(success_rate_file, 'w', encoding='utf-8') as rf:
        json.dump(avg_success_rate_data, rf, ensure_ascii=False, indent=4)

    exit(0)

def main():
    set_random_seed(42)
    parser = get_parser()
    args = parser.parse_args()

    # ==================== Load token embeddings ====================
    with open("./data/rantext_json/cl100_embeddings.json", 'r') as f:
        # cl100_emb is a dictionary, key is the word, value is the corresponding vector
        # Display loading progress
        cl100_emb = json.load(f)
        vector_data_json = {k: cl100_emb[k] for k in tqdm.tqdm(list(cl100_emb.keys())[:11000], desc="Loading cl100_embeddings")}
        cl100_emb = None
        token_to_vector_dict = {token: np.array(vector) for token, vector in vector_data_json.items()}
    if not os.path.exists('./data/rantext_json/sorted_cl100_embeddings.json'):  # We mainly calculate the time of addnoise, the loading time is not calculated, just prepare the pre-file
        func.init_func(1.0, token_to_vector_dict)
    with open('./data/rantext_json/sorted_cl100_embeddings.json', 'r') as f1:
        sorted_cl100_emb = json.load(f1)
    with open('./data/rantext_json/sensitivity_of_embeddings.json', 'r') as f:
        sen_emb = torch.tensor(json.load(f), dtype=torch.float64, device=func.device)  # Ensure sen_emb is a PyTorch tensor

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    logger.info(
        "Running epsilon = %s, EmbInver_K: %d, MaskInfer_K: %d" %
        (args.epsilon, args.EmbInver_K, args.MaskInfer_K))
    
    data_dir = f"./data/{args.task}/{args.dataset}/"
    output_dir = f"./NoiseResults_AttackResults/{args.task}/{args.dataset}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.dataset == "ag_news":
        file_name_list = ['test.tsv']
    elif args.dataset == "SemEval":
        file_name_list = ['SemEval.tsv']
    elif args.dataset == "mednli":
        file_name_list = ['mednli.tsv']
    elif args.dataset == "piidocs":
        file_name_list = ['piidocs.tsv']

    for file_name in file_name_list:
        data_file = os.path.join(data_dir, file_name)
        json_file = os.path.join(output_dir, f'rantext_epsilon_{args.epsilon}_EmbInver_{args.EmbInver_K}_MaskInfer_{args.MaskInfer_K}.json')
        csv_file = os.path.join(output_dir, f'rantext_epsilon_{args.epsilon}_EmbInver_{args.EmbInver_K}_MaskInfer_{args.MaskInfer_K}.csv')

        data = read_data(data_file, args.dataset)
        print(f"Loaded {len(data)} documents from {data_file}")
        print(f"data example: {data[0]}")
        perturbed_data = []

        start_time = time.time()
        interrupted = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load BERT model and tokenizer once
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
        bert_model = BertForMaskedLM.from_pretrained(args.bert_model_path)
        bert_model.to(device)
        bert_model.eval()

        def signal_handler(sig, frame):
            nonlocal interrupted
            interrupted = True
            print("Interrupted! Saving current results...")
            save_results(perturbed_data, json_file, csv_file, start_time, args.dataset)
            print("Results saved. Exiting.")
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Display processing progress
        for i in tqdm.tqdm(range(len(data)), desc="Processing documents"):
            if interrupted:
                break
            if args.dataset == "ag_news" or args.dataset == "SemEval" or args.dataset == "piidocs":
                sentence, label = data[i]
            elif args.dataset == "mednli":
                sentence, label, hypothesis = data[i]

            raw_tokens = func.get_tokens(sentence)  # Get all tokens
            
            # sanitized_sent, EmbInver_success_rate, EmbExpStop_success_rate, EmbInver_success_words, EmbExpStop_success_words, mask_rate, Mask_success_words, mask_rate_ExpStop, Mask_Expstop_success_words
            perturbed_tokens, EmbInver_success_rate, EmbExpStop_success_rate, EmbInver_success_words, EmbExpStop_success_words, mask_rate, Mask_success_words, mask_rate_ExpStop, Mask_Expstop_success_words = func.perturb_sentence(
                args, raw_tokens,
                token_to_vector_dict, sorted_cl100_emb, sen_emb, bert_tokenizer, bert_model, device
            )  # sen_emb is delta_f_new

            if args.dataset == "ag_news" or args.dataset == "SemEval" or args.dataset == "piidocs":
                perturbed_data.append([sentence, label, perturbed_tokens, EmbInver_success_rate, EmbExpStop_success_rate, EmbInver_success_words, EmbExpStop_success_words,
                                       mask_rate, mask_rate_ExpStop, Mask_success_words, Mask_Expstop_success_words])
            elif args.dataset == "mednli":
                perturbed_data.append([sentence, label, hypothesis, perturbed_tokens, EmbInver_success_rate, EmbExpStop_success_rate, EmbInver_success_words, EmbExpStop_success_words,
                                       mask_rate, mask_rate_ExpStop, Mask_success_words, Mask_Expstop_success_words])

        if not interrupted:
            print("Saving final results...")
            save_results(perturbed_data, json_file, csv_file, start_time, args.dataset)

if __name__ == "__main__":
    main()