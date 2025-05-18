import csv
import json
import os
import signal
import random
import time
import numpy as np
import tqdm  # Using import tqdm instead of from tqdm import tqdm
import torch  # Need to import torch
from args_rantext import *
import func_rantext as func

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



def read_data(file_path, dataset):
    data = []
    # Display reading and loading progress

    # Add processing for samsum dataset
    if dataset == "samsum":
        with open(file_path, 'r', encoding='utf-8') as file:
            # JSON format processing
            samsum_data = json.load(file)
            for item in tqdm.tqdm(samsum_data, desc=f"Loading {file_path}"):
                data.append((item["dialogue"], item["summary"]))
        return data

    with open(file_path, 'r', encoding='utf-8') as file:
        header = next(file)
        for line in tqdm.tqdm(file, desc=f"Loading {file_path}"):
            if dataset == "ag_news":
                label, sentence = line.strip().split('\t')
                data.append((sentence, label.strip()))
            elif dataset == "piidocs":
                label, sentence = line.strip().split('\t')
                data.append((sentence, label.strip()))
    return data

def save_results(perturbed_data, json_file, csv_file, start_time, dataset):
    # Custom JSON field names
    if dataset == "ag_news" or dataset == "piidocs":
        json_data = [
            {
                "text": sentence,
                "ground_truth": label,
                "perturbed_sentence": perturbed_tokens
            }
            for sentence, label, perturbed_tokens in perturbed_data
        ]

    # Add processing for samsum dataset
    elif dataset == "samsum":
        json_data = [
            {
                "text": dialogue,  
                "ground_truth": summary,
                "perturbed_sentence": perturbed_dialogue
            }
            for dialogue, summary, perturbed_dialogue in perturbed_data
        ]


    with open(json_file, 'w', encoding='utf-8') as jf:
        json.dump(json_data, jf, ensure_ascii=False, indent=4)

    with open(csv_file, 'w', encoding='utf-8', newline='') as cf:
        writer = csv.writer(cf)
        if dataset == "ag_news" or dataset == "piidocs":
            writer.writerow(['text', 'ground_truth', 'perturbed_sentence'])
        
        elif dataset == "samsum":
            writer.writerow(['text', 'ground_truth', 'perturbed_sentence'])

        
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

    #Speed up exit
    exit(0)

def main():
    set_random_seed(42)
    parser = get_parser()
    args = parser.parse_args()

    # ==================== Load token embeddings ====================
    with open("./data/rantext_json/cl100_embeddings.json", 'r') as f:
        # cl100_emb is a dictionary, key is the word, value is the corresponding vector
        # Display reading and loading progress
        cl100_emb = json.load(f)
        vector_data_json = {k: cl100_emb[k] for k in tqdm.tqdm(list(cl100_emb.keys())[:11000], desc="Loading cl100_embeddings")}
        cl100_emb = None
        token_to_vector_dict = {token: np.array(vector) for token, vector in vector_data_json.items()}
    if not os.path.exists('./data/rantext_json/sorted_cl100_embeddings.json'):##We mainly calculate the addnoise time, so we don't calculate the loading time, just prepare the files directly
        func.init_func(1.0, token_to_vector_dict)
    with open('./data/rantext_json/sorted_cl100_embeddings.json', 'r') as f1:
        sorted_cl100_emb = json.load(f1)
    with open('./data/rantext_json/sensitivity_of_embeddings.json', 'r') as f:
        sen_emb = torch.tensor(json.load(f), dtype=torch.float64, device=func.device)  # Ensure sen_emb is a PyTorch tensor

    data_dir = f"./data/{args.task}/{args.dataset}/"
    output_dir = f"./NoiseResults/{args.task}/{args.dataset}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.dataset == "ag_news":
        file_name_list = ['test.tsv']
    elif args.dataset == "piidocs":
        file_name_list = ['piidocs.tsv']
    # Add new dataset
    elif args.dataset == "samsum":
        file_name_list = ['samsum_combined.json']
    

    for file_name in file_name_list:
        data_file = os.path.join(data_dir, file_name)
        json_file = os.path.join(output_dir, f'rantext_epsilon_{args.epsilon}.json')
        csv_file = os.path.join(output_dir, f'rantext_epsilon_{args.epsilon}.csv')

        data = read_data(data_file, args.dataset)
        print(f"Loaded {len(data)} documents from {data_file}")
        print(f"data example: {data[0]}")
        perturbed_data = []

        start_time = time.time()
        interrupted = False

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

            if args.dataset == "ag_news" or args.dataset == "piidocs":
                sentence, label = data[i]
                raw_tokens = func.get_tokens(sentence)
                perturbed_tokens = func.perturb_sentence(
                    raw_tokens, args.epsilon, args.model,
                    token_to_vector_dict, sorted_cl100_emb, sen_emb
                )
                perturbed_data.append([sentence, label, perturbed_tokens])
            
            
            # Add processing for samsum dataset
            elif args.dataset == "samsum":
                dialogue, summary = data[i]
                raw_tokens = func.get_tokens(dialogue)
                perturbed_tokens = func.perturb_sentence(
                    raw_tokens, args.epsilon, args.model,
                    token_to_vector_dict, sorted_cl100_emb, sen_emb
                )
                perturbed_data.append([dialogue, summary, perturbed_tokens])
            
        if not interrupted:
            print("Saving final results...")
            save_results(perturbed_data, json_file, csv_file, start_time, args.dataset)

if __name__ == "__main__":
    main()