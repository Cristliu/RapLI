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
from utils import *
from spacy.lang.en import English
from transformers import BertTokenizer
from func_SanText import SanText_plus
from args_santext import *

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

def save_results(results, labels, docs, json_file, csv_file, start_time, hypothesis_list=None, endings_info_list=None):
    data = []
    for i, (predicted_text, doc) in enumerate(zip(results, docs)):
        row = {
            "text": " ".join(doc),
            "ground_truth": labels[i],
            "perturbed_sentence": predicted_text
        }
        if hypothesis_list:
            row["hypothesis"] = hypothesis_list[i]
        
        # If it's StoryCloze dataset and ending information is provided
        if endings_info_list and i < len(endings_info_list):
            row.update(endings_info_list[i])

        data.append(row)

    # Save as CSV
    fieldnames = ["text", "ground_truth", "perturbed_sentence"]
    if hypothesis_list:
        fieldnames.append("hypothesis")
    if endings_info_list:
        fieldnames.extend(["ending1", "ending2", "correct_ending"])

    # Save as JSON
    with open(json_file, 'w', encoding='utf-8') as jf:
        json.dump(data, jf, ensure_ascii=False, indent=4)

    with open(csv_file, 'w', encoding='utf-8', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    # Save runtime to JSON
    end_time = time.time()
    runtime = end_time - start_time
    avg_runtime_data = {
        "avg_runtime_seconds": runtime / len(docs),
    }
    runtime_file = json_file.replace('.json', '_avg_addnoise_time.json')
    with open(runtime_file, 'w', encoding='utf-8') as rf:
        json.dump(avg_runtime_data, rf, ensure_ascii=False, indent=4)


def main():
    import json  # Import inside function
    set_random_seed(42)
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger(__name__)
    logger.info("Running dataset: %s,  epsilon = %s" % (
     args.dataset, args.epsilon))

    Embedding_path = os.path.join(args.data_dir, "embeddings")
    args.data_dir = os.path.join(args.data_dir, args.task, args.dataset)

    args.output_dir = os.path.join(args.output_dir, args.task, args.dataset)
    json_file = os.path.join(args.output_dir, f"santext_epsilon_{args.epsilon}.json")
    csv_file = os.path.join(args.output_dir, f"santext_epsilon_{args.epsilon}.csv")

    time_dir = os.path.join(args.output_dir, "Time")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(time_dir):
        os.makedirs(time_dir)

    logger.info("Building Vocabulary...")

    if args.embedding_type=="glove":
        tokenizer = English()
        tokenizer_type="word"
    else:
        tokenizer  = BertTokenizer.from_pretrained(args.bert_model_path)
        tokenizer_type = "subword"
    
    # 'ag_news', 
    if args.dataset == "ag_news":
        vocab = get_vocab_agnews(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)#Although get_vocab_agnews is called for splitting, tokenization is still done directly with English()
    elif args.dataset == "piidocs":
        vocab = get_vocab_piidocs(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    # Add new dataset
    elif args.dataset == "samsum":
        vocab = get_vocab_samsum(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)


    # Print data examples
    logger.info("Vocabulary Example: %s" % str(vocab.most_common(10)))

    sensitive_word_count = int(args.sensitive_word_percentage * len(vocab))
    words = [key for key, _ in vocab.most_common()]
    sensitive_words = words[-sensitive_word_count - 1:]#Take the last sensitive_word_count words as sensitive words

    sensitive_words2id = {word: k for k, word in enumerate(sensitive_words)}
    logger.info("#Total Words: %d, #Sensitive Words: %d" % (len(words),len(sensitive_words2id)))

    sensitive_word_embed = []
    all_word_embed=[]

    word2id = {}
    sword2id = {}
    sensitive_count = 0
    all_count = 0
    
    # if args.embedding_type == "glove":
    #First check if pre-loaded npy files exist
    npy_path = os.path.join(Embedding_path,  f"santext_{args.dataset}_glove_840B-300d.npy")
    word2idx_path = os.path.join(Embedding_path,  f"santext_{args.dataset}_glove_840B-300d_word2idx.npy")
    sword2idx_path = os.path.join(Embedding_path, f"santext_{args.dataset}_glove_840B-300d_sword2idx.npy")

    if os.path.exists(npy_path) and os.path.exists(word2idx_path) and os.path.exists(sword2idx_path):
        all_word_embed = np.load(npy_path)
        word2id = np.load(word2idx_path, allow_pickle=True).item()
        sword2id = np.load(sword2idx_path, allow_pickle=True).item()
        sensitive_word_embed = all_word_embed[list(sword2id.values())]
        #Calculate all_count and sensitive_count
        all_count = len(word2id)#all_count: 125258
        print(f"all_count: {all_count}")
        sensitive_count = len(sword2id)
        logger.info("Loading Word Embedding NPY File: %s" % npy_path)
    else:###Need to track time for building word2id and sword2id
        start_time = time.time()
        num_lines = sum(1 for _ in open(args.word_embedding_path, 'r', encoding='utf-8'))
        logger.info("Loading Word Embedding File: %s" % args.word_embedding_path)
        with open(args.word_embedding_path, 'r', encoding='utf-8') as f:
            # Skip first line if of form count/dim.
            line = f.readline().rstrip().split(' ')
            if len(line) != 2:
                f.seek(0)
            for row in tqdm(f,desc="Loading embeddings", total=num_lines - 1):
                content = row.rstrip().split(' ')
                cur_word=word_normalize(content[0])
                if cur_word in vocab and cur_word not in word2id:
                    word2id[cur_word] = all_count
                    all_count += 1
                    emb=[float(i) for i in content[1:]]
                    all_word_embed.append(emb)
                    if cur_word in sensitive_words2id:
                        sword2id[cur_word] = sensitive_count
                        sensitive_count += 1
                        sensitive_word_embed.append(emb)
                assert len(word2id)==len(all_word_embed)
                assert len(sword2id) == len(sensitive_word_embed)
            f.close()
        end_time = time.time()
        logger.info(f"Building word2id and sword2id time: {end_time - start_time} seconds")###Pretty fast, completes in one minute
        #Save this time to a json file
        time_data = {
            "Building word2id and sword2id time for": end_time - start_time
        }

        time_file = os.path.join(time_dir, f"santext_{args.dataset}_word2id_and_sword2id_time.json")
        with open(time_file, 'w', encoding='utf-8') as tf:
            json.dump(time_data, tf, ensure_ascii=False, indent=4)

        all_word_embed=np.array(all_word_embed, dtype='f')
        sensitive_word_embed = np.array(sensitive_word_embed, dtype='f')

        #Save word2id, sword2id, all_word_embed files
        np.save(npy_path, all_word_embed)
        np.save(word2idx_path, word2id)
        np.save(sword2idx_path, sword2id)
        

    logger.info("All Word Embedding Matrix: %s" % str(all_word_embed.shape))
    logger.info("Sensitive Word Embedding Matrix: %s" % str(sensitive_word_embed.shape))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_word_embed = torch.tensor(all_word_embed).to(device)

    id2word = {v: k for k, v in word2id.items()}  # Build id2word dictionary

    if args.dataset == "ag_news":
        file_name_list = ['test.tsv']
    elif args.dataset == "piidocs":
        file_name_list = ['piidocs.tsv']
    # Add new dataset
    elif args.dataset == "samsum":
        file_name_list = ['samsum_combined.json']
    
    for file_name in file_name_list:
        data_file = os.path.join(args.data_dir, file_name)
        logger.info("Processing file: %s. Will write to: %s and %s" % (data_file, json_file, csv_file))

        num_lines = sum(1 for _ in open(data_file, 'r', encoding='utf-8'))
        with open(data_file, 'r', encoding='utf-8') as rf:
            # header
            header = next(rf)
            labels = []
            docs = []
            hypothesis_list = []
            endings_info_list = []  # Add this line to store StoryCloze extra information

            if args.dataset == "ag_news":
                for line in tqdm(rf, total=num_lines - 1):
                    content = line.strip().split("\t")
                    text = content[1] # index 1 is the content
                    label = int(content[0]) # index 0 is the label
                    doc = [token.text for token in tokenizer(text)]
                    docs.append(doc)
                    labels.append(label)


            elif args.dataset == "piidocs":
                for line in tqdm(rf, total=num_lines - 1):
                    content = line.strip().split("\t")
                    text = content[1]
                    label = int(content[0])
                    doc = [token.text for token in tokenizer(text)]
                    docs.append(doc)
                    labels.append(label)

            # Add after other data processing logic
            # Process SAMSUM dataset
            elif args.dataset == "samsum":
                # JSON file needs special handling
                import json
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for item in tqdm(data, total=len(data)):
                    text = item["dialogue"]
                    label = item["summary"]
                    doc = [token.text for token in tokenizer(text)]
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
                save_results(results, labels, docs, json_file, csv_file, start_time, None, endings_info_list)
            else:
                save_results(results, labels, docs, json_file, csv_file, start_time, hypothesis_list if args.dataset == "mednli" else None)
            logger.info("Results saved. Exiting.")
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        for i, doc in enumerate(tqdm(docs, desc="Sanitize docs using SanText_plus")):
            if interrupted:
                break
            sanitized_doc = SanText_plus(doc, all_word_embed, word2id, id2word, sword2id, words, args.p, args.epsilon, device)
            results.append(sanitized_doc)

        if not interrupted:
            logger.info("Saving ...")
            if args.dataset == "storycloze":
                save_results(results, labels, docs, json_file, csv_file, start_time, None, endings_info_list)
            else:
                save_results(results, labels, docs, json_file, csv_file, start_time, hypothesis_list if args.dataset == "mednli" else None)


if __name__ == "__main__":
    main()