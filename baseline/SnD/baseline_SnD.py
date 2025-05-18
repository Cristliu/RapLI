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
from transformers import BertTokenizer, BertModel
from func_SnD import SnD
from args_SnD import *

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

def save_results(results, labels, ori_texts, json_file, csv_file, start_time, hypothesis_list=None, endings_info_list=None):
    data = []
    for i, (predicted_text, text) in enumerate(zip(results, ori_texts)):
        row = {
            "text": text,
            "ground_truth": labels[i],
            "perturbed_sentence": predicted_text
        }
        if hypothesis_list and i < len(hypothesis_list):
            row["hypothesis"] = hypothesis_list[i]
        
        # Add processing for StoryCloze special fields
        if endings_info_list and i < len(endings_info_list):
            row.update(endings_info_list[i])
            
        data.append(row)

    # Save as JSON
    with open(json_file, 'w', encoding='utf-8') as jf:
        json.dump(data, jf, ensure_ascii=False, indent=4)

    # Save as CSV
    fieldnames = ["text", "ground_truth", "perturbed_sentence"]
    if hypothesis_list:
        fieldnames.append("hypothesis")
    if endings_info_list:
        fieldnames.extend(["ending1", "ending2", "correct_ending"])
        
    with open(csv_file, 'w', encoding='utf-8', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    # Save runtime to JSON
    end_time = time.time()
    runtime = end_time - start_time
    avg_runtime_data = {
        "avg_runtime_seconds": runtime / len(ori_texts),
    }
    runtime_file = json_file.replace('.json', '_avg_addnoise_time.json')
    with open(runtime_file, 'w', encoding='utf-8') as rf:
        json.dump(avg_runtime_data, rf, ensure_ascii=False, indent=4)

def main():
    set_random_seed(42)
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(
        format="%(asctime)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger(__name__)
    logger.info("Running dataset: %s,  epsilon = %s" % (
     args.dataset, args.epsilon))

    data_dir = os.path.join(args.data_dir, args.task, args.dataset)
    output_dir = os.path.join(args.output_dir, args.task, args.dataset)
    json_file = os.path.join(output_dir, f"SnD_epsilon_{args.epsilon}.json")
    csv_file = os.path.join(output_dir, f"SnD_epsilon_{args.epsilon}.csv")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Building Vocabulary...")

#Use bert as tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    model.eval()

    vocab = tokenizer.get_vocab()
    words = list(vocab.keys())
    logger.info("Vocabulary Example: %s" % str(words[:10]))

    if args.dataset == "ag_news":
        file_name_list = ['test.tsv']
    elif args.dataset == "piidocs":
        file_name_list = ['piidocs.tsv']
    # Add new dataset
    elif args.dataset == "samsum":
        file_name_list = ['samsum_combined.json']
    
    for file_name in file_name_list:
        data_file = os.path.join(data_dir, file_name)
        logger.info("Processing file: %s. Will write to: %s and %s" % (data_file, json_file, csv_file))

        num_lines = sum(1 for _ in open(data_file, 'r', encoding='utf-8'))
        with open(data_file, 'r', encoding='utf-8') as rf:
            labels = []
            docs = []
            ori_texts = []
            hypothesis_list = []
            endings_info_list = []  # For storing additional ending information

            # Different file formats need different processing methods
            if args.dataset == "samsum":
                # JSON format processing
                import json
                samsum_data = json.load(rf)
                for item in tqdm(samsum_data):
                    text = item["dialogue"]
                    ori_texts.append(text)
                    label = item["summary"]  # Summary as label
                    doc = tokenizer.tokenize(text)
                    docs.append(doc)
                    labels.append(label)       

            else:
                header = next(rf)
                
                if args.dataset == "ag_news":
                    for line in tqdm(rf, total=num_lines - 1):
                        content = line.strip().split("\t")
                        text = content[1] # index 1 is the content
                        ori_texts.append(text)
                        label = int(content[0]) # index 0 is the label
                        doc = tokenizer.tokenize(text)
                        docs.append(doc)
                        labels.append(label)
            

                elif args.dataset == "piidocs":
                    for line in tqdm(rf, total=num_lines - 1):
                        content = line.strip().split("\t")
                        text = content[1]
                        ori_texts.append(text)
                        label = int(content[0])
                        doc = tokenizer.tokenize(text)
                        docs.append(doc)
                        labels.append(label)

            rf.close()

        results = []
        interrupted = False

        start_time = time.time()

        def signal_handler(sig, frame):
            nonlocal interrupted
            interrupted = True
            logger.info("Interrupted! Saving current results...")
            if args.dataset == "storycloze":
                save_results(results, labels, ori_texts, json_file, csv_file, start_time, None, endings_info_list)
            else:
                save_results(results, labels, ori_texts, json_file, csv_file, start_time, hypothesis_list if args.dataset == "mednli" else None)
            
            logger.info("Results saved. Exiting.")
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        for i, doc in enumerate(tqdm(docs, desc="Sanitize docs using SnD")):#doc is a sentence
            if interrupted:
                break
            perturbed_doc = SnD(doc, tokenizer, model, args.epsilon,device)
            results.append(perturbed_doc)

        if not interrupted:
            logger.info("Saving ...")
            if args.dataset == "storycloze":
                save_results(results, labels, ori_texts, json_file, csv_file, start_time, None, endings_info_list)
            else:
                save_results(results, labels, ori_texts, json_file, csv_file, start_time, hypothesis_list if args.dataset == "mednli" else None)

if __name__ == "__main__":
    main()