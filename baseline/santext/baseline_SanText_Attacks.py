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
from transformers import BertTokenizer, BertForMaskedLM
from func_SanText_Attacks import SanText_plus
from nltk.corpus import stopwords
import string
from args_santext import *


def set_random_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True


def save_results(results, embInver_rate_list, embExpStop_rate_list ,mask_rate_list,mask_rate_ExpStop_list,
                EmbIver_success_words_list, EmbExpStop_success_words_list, Mask_success_words_list, Mask_Expstop_success_words_list,
                labels, docs, json_file, csv_file, start_time, hypothesis_list=None):
    data = []
    for i, (predicted_text, doc, embInver_rate, embExpStop_rate, mask_rate, mask_rate_ExpStop,
            EmbIver_success_words, EmbExpStop_success_words , Mask_success_words, Mask_Expstop_success_words ) in enumerate(
                zip(results, docs, embInver_rate_list,embExpStop_rate_list, mask_rate_list,mask_rate_ExpStop_list,
                    EmbIver_success_words_list, EmbExpStop_success_words_list, Mask_success_words_list, Mask_Expstop_success_words_list)):
        row = {
            "text": " ".join(doc),
            "ground_truth": labels[i],
            "perturbed_sentence": predicted_text,
            "embInver_rate": embInver_rate,
            "embExpStop_rate": embExpStop_rate,
            "embInver_success_words": EmbIver_success_words,
            "embExpStop_success_words": EmbExpStop_success_words,
            "mask_rate": mask_rate,
            "mask_rate_ExpStop": mask_rate_ExpStop,
            "mask_success_words": Mask_success_words,
            "mask_Expstop_success_words": Mask_Expstop_success_words
        }
        if hypothesis_list:
            row["hypothesis"] = hypothesis_list[i]
        data.append(row)

    # Save as JSON
    with open(json_file, 'w', encoding='utf-8') as jf:
        json.dump(data, jf, ensure_ascii=False, indent=4)

    # Save as CSV
    fieldnames = ["text", "ground_truth", "perturbed_sentence",
                  "embInver_rate", "embExpStop_rate" , "embInver_success_words", "embExpStop_success_words",
                  "mask_rate","mask_rate_ExpStop", "mask_success_words", "mask_Expstop_success_words"]
    if hypothesis_list:
        fieldnames.append("hypothesis")
    with open(csv_file, 'w', encoding='utf-8', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    # Save avg_success_rate to JSON
    avg_embInver_rate = np.mean(embInver_rate_list)
    avg_embExpStop_rate = np.mean(embExpStop_rate_list)
    avg_mask_rate = np.mean(mask_rate_list)
    avg_mask_rate_ExpStop = np.mean(mask_rate_ExpStop_list)

    avg_success_rate_data = {
        "avg_embInver_rate": avg_embInver_rate,
        "avg_embExpStop_rate": avg_embExpStop_rate,
        "avg_mask_rate": avg_mask_rate,
        "avg_mask_rate_ExpStop": avg_mask_rate_ExpStop,
    }
    success_rate_file = json_file.replace('.json',
                                          '_avg_success_rate.json')
    with open(success_rate_file, 'w', encoding='utf-8') as rf:
        json.dump(avg_success_rate_data, rf, ensure_ascii=False, indent=4)

    # Save average runtime to JSON
    end_time = time.time()
    runtime = end_time - start_time
    avg_runtime_data = {
        "avg_runtime_seconds": runtime / len(docs) if len(docs) > 0 else 0.0,
    }
    runtime_file = json_file.replace('.json', '_avg_addnoise_with_attack_time.json')
    with open(runtime_file, 'w', encoding='utf-8') as rf:
        json.dump(avg_runtime_data, rf, ensure_ascii=False, indent=4)


def load_embeddings(args, Embedding_path, vocab, sensitive_words2id, device):
    npy_path = os.path.join(Embedding_path,  f"santext_{args.dataset}_glove_840B-300d.npy")
    word2idx_path = os.path.join(Embedding_path,  f"santext_{args.dataset}_glove_840B-300d_word2idx.npy")
    sword2idx_path = os.path.join(Embedding_path, f"santext_{args.dataset}_glove_840B-300d_sword2idx.npy")

    if os.path.exists(npy_path) and os.path.exists(word2idx_path) and os.path.exists(sword2idx_path):
        all_word_embed = torch.tensor(
            np.load(npy_path), dtype=torch.float32).to(device)
        word2id = np.load(word2idx_path, allow_pickle=True).item()
        sword2id = np.load(sword2idx_path, allow_pickle=True).item()
        sensitive_word_embed = all_word_embed[list(sword2id.values())]
        all_count = len(word2id)
        sensitive_count = len(sword2id)
        logging.info(
            f"Loaded preprocessed embeddings. All words: {all_count}, Sensitive words: {sensitive_count}"
        )
    else:
        #print warning
        logging.warning(
            "Preprocessed embeddings not found. Loading raw embeddings and processing..."
        )
        all_word_embed = []
        word2id = {}
        sword2id = {}
        sensitive_word_embed = []
        sensitive_count = 0
        all_count = 0

        num_lines = sum(1 for _ in open(args.word_embedding_path, 'r', encoding='utf-8'))
        logging.info("Loading Word Embedding File: %s" %
                     args.word_embedding_path)

        with open(args.word_embedding_path, 'r', encoding='utf-8') as f:
            # Skip first line if of form count/dim.
            line = f.readline().rstrip().split(' ')
            if len(line) != 2:
                f.seek(0)
            for row in tqdm(f,desc="Loading embeddings", total=num_lines - 1):
                content = row.rstrip().split(' ')
                cur_word = word_normalize(content[0])
                if cur_word in vocab and cur_word not in word2id:
                    word2id[cur_word] = all_count
                    all_count += 1
                    emb = [float(i) for i in content[1:]]
                    all_word_embed.append(emb)
                    if cur_word in sensitive_words2id:
                        sword2id[cur_word] = sensitive_count
                        sensitive_count += 1
                        sensitive_word_embed.append(emb)
                assert len(word2id)==len(all_word_embed)
                assert len(sword2id) == len(sensitive_word_embed)
            f.close()

        all_word_embed=np.array(all_word_embed, dtype='f')
        sensitive_word_embed = np.array(sensitive_word_embed, dtype='f')
        
        # save word2id，sword2id，all_word_embed
        np.save(npy_path, all_word_embed)
        np.save(word2idx_path, word2id)
        np.save(sword2idx_path, sword2id)

        logging.info(
            f"Saved processed embeddings. All words: {all_count}, Sensitive words: {sensitive_count}"
        )

    return all_word_embed, word2id, sword2id, sensitive_word_embed


def build_vocab(args, tokenizer, tokenizer_type, dataset):
    # 'ag_news', "SemEval", "mednli"
    if args.dataset == "ag_news":
        vocab = get_vocab_agnews(args.data_dir, tokenizer, tokenizer_type=tokenizer_type) # Although get_vocab_agnews is called here according to split, it is still directly tokenized by English() below
    elif args.dataset == "SemEval":
        vocab = get_vocab_SemEval(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    elif args.dataset == "mednli":
        vocab = get_vocab_mednli(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    elif args.dataset == "piidocs":
        vocab = get_vocab_piidocs(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    return vocab


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
    logger.info("Running dataset: %s,  epsilon = %s" % (
     args.dataset, args.epsilon))
    
    Embedding_path = os.path.join(args.data_dir, "embeddings")
    args.data_dir = os.path.join(args.data_dir, args.task, args.dataset)
    args.attack_output_dir = os.path.join(args.attack_output_dir, args.task, args.dataset)
    json_file = os.path.join(args.attack_output_dir, f"santext_epsilon_{args.epsilon}_EmbInver_{args.EmbInver_K}_MaskInfer_{args.MaskInfer_K}.json")
    csv_file = os.path.join(args.attack_output_dir, f"santext_epsilon_{args.epsilon}_EmbInver_{args.EmbInver_K}_MaskInfer_{args.MaskInfer_K}.csv")

    if not os.path.exists(args.attack_output_dir):
        os.makedirs(args.attack_output_dir)

    logger.info("Building Vocabulary...")

    if args.embedding_type == "glove":
        tokenizer = English()
        tokenizer_type = "word"
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
        tokenizer_type = "subword"

    vocab = build_vocab(args, tokenizer, tokenizer_type, args.dataset)

    # Print data sample
    logger.info("Vocabulary Example: %s" % str(vocab.most_common(10)))

    sensitive_word_count = int(args.sensitive_word_percentage * len(vocab))
    words = [key for key, _ in vocab.most_common()]
    sensitive_words = words[-sensitive_word_count - 1:] # Take the last sensitive_word_count words as sensitive words

    # vocab.sensitive_words = set(sensitive_words)
    sensitive_words2id = {word: idx for idx, word in enumerate(sensitive_words)}
    logger.info("#Total Words: %d, #Sensitive Words: %d" % (len(words), len(sensitive_words2id)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load embeddings
    all_word_embed, word2id, sword2id, sensitive_word_embed = load_embeddings(
        args, Embedding_path, vocab, sensitive_words2id, device)

    id2word = {v: k for k, v in word2id.items()}

    # Determine which files to process based on the dataset
    if args.dataset == "ag_news":
        file_name_list = ['test.tsv']
    elif args.dataset == "SemEval":
        file_name_list = ['SemEval.tsv']
    elif args.dataset == "mednli":
        file_name_list = ['mednli.tsv']
    elif args.dataset == "piidocs":
        file_name_list = ['piidocs.tsv']

    stop_words = set(stopwords.words('english'))
    punctuation_token = set(string.punctuation)
    # print(f"punctuation_token: {punctuation_token}")
    stop_words = stop_words.union(punctuation_token)

    for file_name in file_name_list:
        data_file = os.path.join(args.data_dir, file_name)
        logger.info(
            f"Processing file: {data_file}. Will write to: {json_file} and {csv_file}"
        )

        num_lines = sum(1 for _ in open(data_file, 'r', encoding='utf-8'))
        with open(data_file, 'r', encoding='utf-8') as rf:
            # header
            header = next(rf)
            labels = []
            docs = []
            hypothesis_list = []

            if args.dataset == "ag_news":
                for line in tqdm(rf, total=num_lines - 1):
                    content = line.strip().split("\t")
                    text = content[1] # index 1 is the content
                    label = int(content[0]) # index 0 is the label
                    doc = [token.text for token in tokenizer(text)]
                    docs.append(doc)
                    labels.append(label)
            
            elif args.dataset == "SemEval":
                for line in tqdm(rf, total=num_lines - 1):
                    content = line.strip().split("\t")
                    text = content[2]
                    label = int(content[1])
                    doc = [token.text for token in tokenizer(text)]
                    docs.append(doc)
                    labels.append(label)


            elif args.dataset == "mednli":
                for line in tqdm(rf, total=num_lines - 1):
                    content = line.strip().split("\t")
                    text = content[1]
                    label = int(content[0])
                    # hypothesis = content[2] and remove leading spaces
                    hypothesis = content[2].lstrip() ###- To be consistent with other methods, it is not removed for now
                    doc = [token.text for token in tokenizer(text)]
                    docs.append(doc)
                    labels.append(label)
                    hypothesis_list.append(hypothesis)

            elif args.dataset == "piidocs":
                for line in tqdm(rf, total=num_lines - 1):
                    content = line.strip().split("\t")
                    text = content[1]
                    label = int(content[0])
                    doc = [token.text for token in tokenizer(text)]
                    docs.append(doc)
                    labels.append(label)

        # Initialize SanText_plus with loaded model and tokenizer if needed
        sanitized_docs = []
        embInver_rate_list = []
        embExpStop_rate_list = []
        mask_rate_list = []
        mask_rate_ExpStop_list = []
        EmbIver_success_words_list = []
        EmbExpStop_success_words_list = []
        Mask_success_words_list = []
        Mask_Expstop_success_words_list = []

        interrupted = False

        start_time = time.time()

        def signal_handler(sig, frame):
            nonlocal interrupted
            interrupted = True
            logger.info("Interrupted! Saving current results...")
            save_results(results=sanitized_docs,
                         embInver_rate_list=embInver_rate_list,
                         embExpStop_rate_list=embExpStop_rate_list,
                         mask_rate_list=mask_rate_list,
                         mask_rate_ExpStop_list=mask_rate_ExpStop_list,
                         EmbIver_success_words_list=EmbIver_success_words_list,
                         EmbExpStop_success_words_list=EmbExpStop_success_words_list,
                         Mask_success_words_list=Mask_success_words_list,
                         Mask_Expstop_success_words_list=Mask_Expstop_success_words_list,
                         labels=labels,
                         docs=docs,
                         json_file=json_file,
                         csv_file=csv_file,
                         start_time=start_time,
                         hypothesis_list = hypothesis_list if args.dataset == "mednli" else None)
            logger.info("Results saved. Exiting.")
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Load BERT model and tokenizer once
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
        bert_model = BertForMaskedLM.from_pretrained(args.bert_model_path)
        bert_model.to(device)
        bert_model.eval()


        # Process documents
        for i, doc in enumerate(tqdm(docs, desc="Sanitizing documents")):
            if interrupted:
                break

            # new_doc_str, embInver_rate, embExpStop_rate, mask_rate, mask_rate_ExpStop, EmbIver_success_words, EmbExpStop_success_words, Mask_success_words, Mask_Expstop_success_words
            sanitized_doc, embInver_rate, embExpStop_rate, mask_rate, mask_rate_ExpStop, EmbIver_success_words, EmbExpStop_success_words, Mask_success_words, Mask_Expstop_success_words = SanText_plus(
                args=args,
                doc=doc,
                embedding_matrix=all_word_embed,
                word2id=word2id,
                id2word=id2word,
                sword2id=sword2id,
                all_words=words,
                p=args.p,
                epsilon=args.epsilon,
                device=device,
                bert_tokenizer=bert_tokenizer,
                bert_model=bert_model,
                EmbInver_K=args.EmbInver_K,
                stop_words=stop_words
            )

            embInver_rate_list.append(embInver_rate)
            embExpStop_rate_list.append(embExpStop_rate)
            mask_rate_list.append(mask_rate)
            mask_rate_ExpStop_list.append(mask_rate_ExpStop)
            EmbIver_success_words_list.append(EmbIver_success_words)
            EmbExpStop_success_words_list.append(EmbExpStop_success_words)
            Mask_success_words_list.append(Mask_success_words)
            Mask_Expstop_success_words_list.append(Mask_Expstop_success_words)
            sanitized_docs.append(sanitized_doc)

        if not interrupted:
            logger.info("Saving final results...")
            save_results(results=sanitized_docs,
                         embInver_rate_list=embInver_rate_list,
                         embExpStop_rate_list=embExpStop_rate_list,
                         mask_rate_list=mask_rate_list,
                         mask_rate_ExpStop_list=mask_rate_ExpStop_list,
                         EmbIver_success_words_list=EmbIver_success_words_list,
                         EmbExpStop_success_words_list=EmbExpStop_success_words_list,
                         Mask_success_words_list=Mask_success_words_list,
                         Mask_Expstop_success_words_list=Mask_Expstop_success_words_list,
                         labels=labels,
                         docs=docs,
                         json_file=json_file,
                         csv_file=csv_file,
                         start_time=start_time,
                         hypothesis_list = hypothesis_list if args.dataset == "mednli" else None)


if __name__ == "__main__":
    main()
