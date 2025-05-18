import string
import torch
import random
import numpy as np
import logging
import os
import signal
import json
import csv
import time
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from func_SnD_Attacks import SnD_plus
from nltk.corpus import stopwords
from args_SnD import *

def set_random_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

def save_results(perturbed_data, json_file, csv_file, start_time, dataset):
    data = []
    for item in perturbed_data:
        row = {
            "original_text": item["original_text"],
            "ground_truth": item["label"],
            "perturbed_sentence": item["perturbed_sentence"],
            "embInver_rate": item["embInver_rate"],
            "embExpStop_rate": item["embExpStop_rate"],
            "embInver_success_words": item["embInver_success_words"],
            "embExpStop_success_words": item["embExpStop_success_words"],
            "mask_rate": item["mask_rate"],
            "mask_rate_ExpStop": item["mask_rate_ExpStop"],
            "mask_success_words": item["mask_success_words"],
            "mask_Expstop_success_words": item["mask_Expstop_success_words"],
        }
        # #2. Only write hypothesis when dataset is mednli
        if dataset == "mednli" and "hypothesis" in item and item["hypothesis"] is not None:
            row["hypothesis"] = item["hypothesis"]
        data.append(row)

    with open(json_file, 'w', encoding='utf-8') as jf:
        json.dump(data, jf, ensure_ascii=False, indent=4)

    fieldnames = [
        "original_text","ground_truth","perturbed_sentence",
        "embInver_rate","embExpStop_rate",
        "embInver_success_words","embExpStop_success_words",
        "mask_rate","mask_rate_ExpStop",
        "mask_success_words","mask_Expstop_success_words",
    ]
    if dataset == "mednli":
        fieldnames.insert(3, "hypothesis")

    with open(csv_file, 'w', encoding='utf-8', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for d in data:
            writer.writerow(d)

    # ----------------------------------------------------------------------------
    # Calculate and save average success rates here
    # ----------------------------------------------------------------------------
    if len(data) > 0:
        embInver_rate_list = [row["embInver_rate"] for row in data]
        embExpStop_rate_list = [row["embExpStop_rate"] for row in data]
        mask_rate_list = [row["mask_rate"] for row in data]
        mask_rate_ExpStop_list = [row["mask_rate_ExpStop"] for row in data]

        avg_embInver_rate = float(np.mean(embInver_rate_list))
        avg_embExpStop_rate = float(np.mean(embExpStop_rate_list))
        avg_mask_rate = float(np.mean(mask_rate_list))
        avg_mask_rate_ExpStop = float(np.mean(mask_rate_ExpStop_list))

        avg_success_rate_data = {
            "avg_embInver_rate": avg_embInver_rate,
            "avg_embExpStop_rate": avg_embExpStop_rate,
            "avg_mask_rate": avg_mask_rate,
            "avg_mask_rate_ExpStop": avg_mask_rate_ExpStop,
        }
        success_rate_file = json_file.replace('.json', '_avg_success_rate.json')
        with open(success_rate_file, 'w', encoding='utf-8') as rf:
            json.dump(avg_success_rate_data, rf, ensure_ascii=False, indent=4)
    # ----------------------------------------------------------------------------




    end_time = time.time()
    runtime = end_time - start_time
    avg_runtime_data = {
        "avg_runtime_seconds": runtime / len(perturbed_data) if perturbed_data else 0.0
    }
    runtime_file = json_file.replace('.json','_avg_addnoise_with_attack_time.json')
    with open(runtime_file, 'w', encoding='utf-8') as rf:
        json.dump(avg_runtime_data, rf, ensure_ascii=False, indent=4)

def main():
    set_random_seed(42)
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    logger.info("Running dataset: %s, epsilon=%s, EmbInver_K=%s, MaskInfer_K=%s" % (
        args.dataset, args.epsilon, args.EmbInver_K, args.MaskInfer_K))

    data_dir = os.path.join(args.data_dir, args.task, args.dataset)
    output_dir = os.path.join(args.attack_output_dir, args.task, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_file = os.path.join(
        output_dir,
        f"SnD_epsilon_{args.epsilon}_EmbInver_{args.EmbInver_K}_MaskInfer_{args.MaskInfer_K}.json"
    )
    csv_file = os.path.join(
        output_dir,
        f"SnD_epsilon_{args.epsilon}_EmbInver_{args.EmbInver_K}_MaskInfer_{args.MaskInfer_K}.csv"
    )

    stop_words = set(stopwords.words('english'))
    punctuation_token = set(string.punctuation)
    stop_words = stop_words.union(punctuation_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_mask_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    bert_model.to(device)
    bert_mask_model.to(device)
    bert_model.eval()
    bert_mask_model.eval()

    if args.dataset == "ag_news":
        file_name_list = ["test.tsv"]
    elif args.dataset == "SemEval":
        file_name_list = ["SemEval.tsv"]
    elif args.dataset == "mednli":
        file_name_list = ["mednli.tsv"]
    elif args.dataset == "piidocs":
        file_name_list = ["piidocs.tsv"]
    else:
        file_name_list = []

    perturbed_data = []
    start_time = time.time()
    interrupted = False

    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        logger.info("Interrupted! Saving current results...")
        save_results(perturbed_data, json_file, csv_file, start_time, args.dataset)
        logger.info("Results saved. Exiting.")
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    for file_name in file_name_list:
        data_file = os.path.join(data_dir, file_name)
        num_lines = sum(1 for _ in open(data_file, 'r', encoding='utf-8'))
        with open(data_file, 'r', encoding='utf-8') as rf:
            header = next(rf)
            for line in tqdm(rf, total=num_lines - 1, desc=f"SnD Attacks on {file_name}"):
                content = line.strip().split("\t")

                # #3. Pass corresponding parameters when calling SnD_plus for different datasets (to avoid index mismatch and None value conflicts)
                if args.dataset == "ag_news":
                    label = int(content[0])
                    text = content[1]
                    hypothesis = None
                elif args.dataset == "SemEval":
                    label = int(content[1])
                    text = content[2]
                    hypothesis = None
                elif args.dataset == "mednli":
                    label = int(content[0])
                    text = content[1]
                    hypothesis = content[2].lstrip()
                elif args.dataset == "piidocs":
                    label = int(content[0])
                    text = content[1]
                    hypothesis = None
                else:
                    label = ""
                    text = ""
                    hypothesis = None

                doc_tokens = bert_tokenizer.tokenize(text)

                item = SnD_plus(
                    args=args,
                    doc_tokens=doc_tokens,
                    original_text=text,
                    label=label,
                    hypothesis=hypothesis,
                    tokenizer=bert_tokenizer,
                    bert_model=bert_model,
                    bert_mask_model=bert_mask_model,
                    stop_words=stop_words,
                    device=device,
                    dataset=args.dataset  # Pass dataset, control internally whether hypothesis is present
                )
                perturbed_data.append(item)

                # return 0

                if interrupted:
                    break
        if interrupted:
            break

    if not interrupted:
        logger.info("Saving final results...")
        save_results(perturbed_data, json_file, csv_file, start_time, args.dataset)

if __name__ == "__main__":
    main()