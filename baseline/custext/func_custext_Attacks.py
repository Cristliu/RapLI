import os
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from collections import Counter, defaultdict
import json
from nltk.corpus import stopwords
import warnings
from args_custext import *
import torch
import time
import unicodedata
import string
from transformers import logging as transformers_logging


warnings.filterwarnings('ignore')

parser = get_parser()
args = parser.parse_args()

stop_words = set(stopwords.words('english'))
# Get all punctuation marks and represent them as set([',', '.']) like form
punctuation_token = set(string.punctuation)
# print(f"punctuation_token: {punctuation_token}")
stop_words = stop_words.union(punctuation_token)


def batch_inference(texts, tokenizer, model, device, top_k=5):
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
        # if original_subtokens[i] in stop_words:
        #     continue
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

def word_normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

def load_data(dataset=None):
    print(f'__loading__{args.dataset}__')
    if args.dataset == "ag_news":
        test_df = pd.read_csv(f"./data/{args.task}/{args.dataset}/test.tsv", sep='\t')
    elif args.dataset == "SemEval":
        test_df = pd.read_csv(f"./data/{args.task}/{args.dataset}/SemEval.tsv", sep='\t')
    elif args.dataset == "mednli":
        test_df = pd.read_csv(f"./data/{args.task}/{args.dataset}/mednli.tsv", sep='\t')
    elif args.dataset == "piidocs":
        test_df = pd.read_csv(f"./data/{args.task}/{args.dataset}/piidocs.tsv", sep='\t')
    return test_df

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def load_embeddings(device):
    npy_path = f'./data/embeddings/custext_glove_840B-300d.npy'
    word2idx_path = f'./data/embeddings/custext_glove_840B-300d_word2idx.npy'

    if os.path.exists(npy_path) and os.path.exists(word2idx_path):
        embeddings = np.load(npy_path)
        word2idx = np.load(word2idx_path, allow_pickle=True).item()
        idx2word = np.array(list(word2idx.keys()))
        embeddings = torch.from_numpy(embeddings).to(device)
    else:
        start_time = time.time()
        embedding_path = f'./data/embeddings/glove_840B-300d.txt'
        embeddings = []
        word2idx = {}
        idx2word = []
        all_count = 0

        num_lines = sum(1 for _ in open(embedding_path, 'r', encoding='utf-8'))

        with open(embedding_path, 'r', encoding='utf-8') as f:
            # Skip first line if of form count/dim.
            line = f.readline().rstrip().split(' ')
            if len(line) != 2:
                f.seek(0)
            for line in tqdm(f, desc="Loading embeddings", total=num_lines - 1):
                content = line.rstrip().split(' ')
                cur_word=word_normalize(content[0])
                ### Here, the vocabulary is not pre-loaded, but all are mapped
                word2idx[cur_word] = all_count
                idx2word.append(cur_word)
                embeddings.append([float(val) for val in content[1:]])
                all_count += 1

        embeddings_np = np.array(embeddings,dtype="f")
        idx2word = np.array(idx2word)

        np.save(npy_path, embeddings_np)
        np.save(word2idx_path, word2idx)

        end_time = time.time()
        print(f"Loading embeddings time: {end_time - start_time} seconds")
        # Save this time to a json file
        time_data = {
            "load_embeddings_time": end_time - start_time
        }
        with open(f"./NoiseResults/{args.task}/{args.dataset}/Time/custext_load_embeddings_time.json", 'w', encoding='utf-8') as time_file:
            json.dump(time_data, time_file, ensure_ascii=False, indent=4)

        # Convert to PyTorch tensor # Otherwise, running this else branch will report an error
        embeddings = torch.from_numpy(embeddings_np).to(device)

    return embeddings, word2idx, idx2word

def get_customized_mapping(eps, top_k):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings, word2idx, idx2word = load_embeddings(device)
    df_test = load_data(args.dataset)
    
    if args.dataset == "ag_news":
        test_corpus = " ".join(df_test.text)
    elif args.dataset == "SemEval":
        test_corpus = " ".join(df_test.tweet)
    elif args.dataset == "mednli":
        test_corpus = " ".join(df_test.sentence1)
    elif args.dataset == "piidocs":
        test_corpus = " ".join(df_test.text)
    
    word_freq = [x[0] for x in Counter(test_corpus.split()).most_common()]
    print(f"Total number of unique words in the corpus: {len(word_freq)}")
    
    sim_word_dict = defaultdict(list)
    p_dict = defaultdict(list)
    
    for i in trange(len(word_freq)):
        word = word_freq[i]
        if word in word2idx:
            word_idx = word2idx[word]
            word_embedding = embeddings[word_idx].to(device)  # Shape: [1, embedding_dim]
            
            distances = torch.norm(embeddings - word_embedding, dim=1)
            
            # Get the indices of the top_k closest words
            sorted_distances, indices = torch.topk(distances, top_k, largest=False)
            word_list = [idx2word[idx] for idx in indices.cpu().numpy()]
            # embedding_list = embeddings[indices]
            
            # Calculate probability distribution
            min_dist = sorted_distances.min()
            max_dist = sorted_distances.max()
            range_dist = max_dist - min_dist
            if range_dist == 0:
                range_dist = 1e-6  # Avoid division by zero
            # Normalize distances    
            new_sim_dist_list = - (sorted_distances - min_dist) / range_dist
            tmp = torch.exp(eps * new_sim_dist_list / 2)
            p = tmp / tmp.sum()
            p_dict[word] = p.cpu().numpy().tolist()

            sim_word_dict[word] = word_list


    p_dict_path = f"./data/custext_dict/p_dict_{args.dataset}_eps_{args.epsilon}.txt"
    sim_word_dict_path = f"./data/custext_dict/sim_word_dict_{args.dataset}_eps_{args.epsilon}.txt"

    # Save the generated dictionaries as JSON files
    try:
        with open(p_dict_path, 'w', encoding='utf-8') as json_file:
            json_file.write(json.dumps(p_dict, ensure_ascii=False, indent=4))
    except IOError:
        pass

    try:
        with open(sim_word_dict_path, 'w', encoding='utf-8') as json_file:
            json_file.write(json.dumps(sim_word_dict, ensure_ascii=False, indent=4))
    except IOError:
        pass

    return sim_word_dict,p_dict

# Find the K nearest words in the embedding space for a single word
def find_K_nearest_words(word,embeddings, word2idx, idx2word,K):
    if word not in word2idx:
        return [word]
    else:
        word_idx = word2idx[word]
        word_embedding = embeddings[word_idx].unsqueeze(0)  # Shape: [1, embedding_dim]
    
        distances = torch.norm(embeddings - word_embedding, dim=1)
        sorted_distances, indices = torch.topk(distances, K, largest=False)
        word_list = [idx2word[idx] for idx in indices.cpu().numpy()]
        return word_list

def generate_new_sents_s1(df, sim_word_dict, p_dict, save_stop_words, type="test", EmbInver_K=1, MaskInfer_K=1, bert_tokenizer=None, bert_model=None):

    cnt = 0
    raw_cnt = 0
    stop_cnt = 0
    K = EmbInver_K
    EmbInver_success_cnt = 0
    EmbExpStop_success_cnt = 0
    attack_cnt = 0
    EmbInver_success_rate_list = []
    EmbExpStop_success_rate_list = []
    EmbInver_success_words_list = []
    EmbExpStop_success_words_list = []

    Mask_success_rate_list = []
    Mask_Expstop_success_rate_list = []
    Mask_success_words_list = []
    Mask_Expstop_success_words_list = []

    if args.dataset == "ag_news":
        dataset = df.text
    elif args.dataset == "SemEval":
        dataset = df.tweet
    elif args.dataset == "mednli":
        dataset = df.sentence1
    elif args.dataset == "piidocs":
        dataset = df.text
    new_dataset = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings, word2idx, idx2word = load_embeddings(device)

    for i in trange(len(dataset)):
        record = dataset[i].split()
        orig_record = []
        new_record = []
        EmbInver_success_words = []
        EmbExpStop_success_words = []
        Mask_success_words = []
        Mask_Expstop_success_words = []

        for word in record:
            orig_record.append(word)
            # print(f"Processing word '{word}'")
            if (save_stop_words and word in stop_words) or (word not in sim_word_dict):
                if word in stop_words:
                    # print(f"Skipping stop word '{word}'")
                    stop_cnt += 1
                    raw_cnt += 1
                if is_number(word):
                    try:
                        word = str(round(float(word)) + np.random.randint(1000))
                    except:
                        pass
                new_record.append(word)

                new_word = word # If it is a stop word or a number, new_word is directly equal to word
                if new_word in sim_word_dict:
                    new_word_p = np.array(p_dict[new_word])
                    # Return the indices of the top K largest p values
                    new_word_p_idx = np.argsort(new_word_p)[::-1][:K]
                    # Return the words corresponding to the top K largest p values
                    new_word_p_word = [sim_word_dict[new_word][i] for i in new_word_p_idx]
                else: # If not, directly select the K nearest words in the embedding space
                    # print(f"new_word {new_word} is not in sim_word_dict")
                    new_word_p_word = find_K_nearest_words(new_word,embeddings, word2idx, idx2word,K)
                

                attack_cnt += 1
                # Compare uniformly in lowercase
                if word.lower() in [x.lower() for x in new_word_p_word]:
                    # print(f"Attack succeeded: {word} -> {new_word}")
                    EmbInver_success_cnt += 1
                    EmbInver_success_words.append(word)
                    if word.lower() not in stop_words:
                        EmbExpStop_success_cnt += 1
                        EmbExpStop_success_words.append(word)
                # else:
                #     print(f"Attack failed: {word} -> {new_word}")

            else:
                p = np.array(p_dict[word])
                if np.isnan(p).any():
                    print(f"Skipping word '{word}' due to NaN in probabilities")
                    new_record.append(word)
                    continue
                # Normalize probability array
                p = p / p.sum()
                new_word = np.random.choice(sim_word_dict[word], p=p)
                
                # Embedding Inversion Attack, check if word is in the top K nearest neighbors of new_word, if so, the attack is successful
                # print(f"\nword: {word}")
                # First get the sim_word_dict of new_word and the corresponding p_dict value, return the top K largest p values
                # Consider that new_word may not be in sim_word_dict, so add judgment; if not, directly select the K nearest words in the embedding space
                if new_word in sim_word_dict:
                    new_word_p = np.array(p_dict[new_word])
                    # Return the indices of the top K largest p values
                    new_word_p_idx = np.argsort(new_word_p)[::-1][:K]
                    # Return the words corresponding to the top K largest p values
                    new_word_p_word = [sim_word_dict[new_word][i] for i in new_word_p_idx]
                else: # If not, directly select the K nearest words in the embedding space
                    # print(f"new_word {new_word} is not in sim_word_dict")
                    new_word_p_word = find_K_nearest_words(new_word,embeddings, word2idx, idx2word,K)
                
                
                # print(f"new_word: {new_word}")
                # print(f"new_word_p_word: {new_word_p_word}")
                attack_cnt += 1
                # Compare uniformly in lowercase
                if word.lower() in [x.lower() for x in new_word_p_word]:
                    # print(f"Attack succeeded: {word} -> {new_word}")
                    EmbInver_success_cnt += 1
                    EmbInver_success_words.append(word)
                    if word.lower() not in stop_words:
                        EmbExpStop_success_cnt += 1
                        EmbExpStop_success_words.append(word)


                # print(f"new_word: {new_word}")
                new_record.append(new_word)
                if new_word == word:
                    raw_cnt += 1
            cnt += 1
        
        # Calculate attack success rate, i.e., success_cnt/attack_cnt
        EmbInver_success_rate = EmbInver_success_cnt / attack_cnt
        EmbExpStop_success_rate = EmbExpStop_success_cnt / attack_cnt

        # Add EmbInver_success_rate and EmbExpStop_success_rate to the list
        EmbInver_success_rate_list.append(EmbInver_success_rate)
        EmbExpStop_success_rate_list.append(EmbExpStop_success_rate)

        EmbInver_success_words_list.append(EmbInver_success_words)
        EmbExpStop_success_words_list.append(EmbExpStop_success_words)

        # Mask Token Inference Attack
        if bert_tokenizer and bert_model:
            mask_rate, Mask_success_words, mask_rate_ExpStop, Mask_Expstop_success_words = mask_token_inference_attack_subword_topk_batch(
                Mask_success_words, Mask_Expstop_success_words, record, new_record, bert_tokenizer, bert_model,
                device, top_k=MaskInfer_K, batch_size=64, stop_words=stop_words
            )
        else:
            mask_rate = 0.0

        Mask_success_rate_list.append(mask_rate)
        Mask_Expstop_success_rate_list.append(mask_rate_ExpStop)
        Mask_success_words_list.append(Mask_success_words)
        Mask_Expstop_success_words_list.append(Mask_Expstop_success_words)
        
        new_dataset.append(" ".join(new_record))

    df['new_content'] = new_dataset
    df['EmbInver_success_rate'] = EmbInver_success_rate_list
    df['EmbExpStop_success_rate'] = EmbExpStop_success_rate_list
    df['EmbInver_success_words'] = EmbInver_success_words_list
    df['EmbExpStop_success_words'] = EmbExpStop_success_words_list
    df['Mask_success_rate'] = Mask_success_rate_list
    df['Mask_Expstop_success_rate'] = Mask_Expstop_success_rate_list
    df['Mask_success_words'] = Mask_success_words_list
    df['Mask_Expstop_success_words'] = Mask_Expstop_success_words_list
    


    output_dir = f"./NoiseResults_AttackResults/{args.task}/{args.dataset}" 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if type == "test":
        json_output_path = f"{output_dir}/custext_epsilon_{args.epsilon}_EmbInver_{args.EmbInver_K}_MaskInfer_{args.MaskInfer_K}.json"
        csv_output_path = f"{output_dir}/custext_epsilon_{args.epsilon}_EmbInver_{args.EmbInver_K}_MaskInfer_{args.MaskInfer_K}.csv"

    if args.dataset == "ag_news":
        df.rename(columns={'label': 'ground_truth', 'text': 'text', 'new_content': 'perturbed_sentence','EmbInver_success_rate':'embInver_rate',
        "EmbExpStop_success_rate":"embExpStop_rate",
        "EmbInver_success_words":"embInver_success_words",
        "EmbExpStop_success_words":"embExpStop_success_words",
        "Mask_success_rate":"mask_rate",
        "Mask_Expstop_success_rate":"mask_rate_ExpStop",
        "Mask_success_words":"mask_success_words",
        "Mask_Expstop_success_words":"mask_Expstop_success_words"
        }, inplace=True)

        perturbed_data = df[['text', 'ground_truth', 'perturbed_sentence',"embInver_rate","embExpStop_rate","embInver_success_words","embExpStop_success_words","mask_rate","mask_rate_ExpStop","mask_success_words","mask_Expstop_success_words"]].to_dict(orient='records')
        df[['text', 'ground_truth', 'perturbed_sentence',"embInver_rate","embExpStop_rate","embInver_success_words","embExpStop_success_words","mask_rate","mask_rate_ExpStop","mask_success_words","mask_Expstop_success_words"]].to_csv(csv_output_path, index=False, encoding='utf-8')

    elif args.dataset == "SemEval":
        df.rename(columns={'sentiment': 'ground_truth', 'tweet': 'text', 'new_content': 'perturbed_sentence','EmbInver_success_rate':'embInver_rate',
        "EmbExpStop_success_rate":"embExpStop_rate",
        "EmbInver_success_words":"embInver_success_words",
        "EmbExpStop_success_words":"embExpStop_success_words",
        "Mask_success_rate":"mask_rate",
        "Mask_Expstop_success_rate":"mask_rate_ExpStop",
        "Mask_success_words":"mask_success_words",
        "Mask_Expstop_success_words":"mask_Expstop_success_words"
        }, inplace=True)
        perturbed_data = df[['text', 'ground_truth', 'perturbed_sentence',"embInver_rate","embExpStop_rate","embInver_success_words","embExpStop_success_words","mask_rate","mask_rate_ExpStop","mask_success_words","mask_Expstop_success_words"]].to_dict(orient='records')
        df[['text', 'ground_truth', 'perturbed_sentence',"embInver_rate","embExpStop_rate","embInver_success_words","embExpStop_success_words","mask_rate","mask_rate_ExpStop","mask_success_words","mask_Expstop_success_words"]].to_csv(csv_output_path, index=False, encoding='utf-8')

    elif args.dataset == "mednli":
        df.rename(columns={'gold_label': 'ground_truth', 'sentence1': 'text', 'new_content': 'perturbed_sentence','sentence2':'hypothesis',
        'EmbInver_success_rate':'embInver_rate',
        "EmbExpStop_success_rate":"embExpStop_rate",
        "EmbInver_success_words":"embInver_success_words",
        "EmbExpStop_success_words":"embExpStop_success_words",
        "Mask_success_rate":"mask_rate",
        "Mask_Expstop_success_rate":"mask_rate_ExpStop",
        "Mask_success_words":"mask_success_words",
        "Mask_Expstop_success_words":"mask_Expstop_success_words"
        }, inplace=True)
        perturbed_data = df[['text', 'ground_truth', 'perturbed_sentence', "hypothesis" ,"embInver_rate","embExpStop_rate","embInver_success_words","embExpStop_success_words","mask_rate","mask_rate_ExpStop","mask_success_words","mask_Expstop_success_words"]].to_dict(orient='records')
        df[['text', 'ground_truth', 'perturbed_sentence', "hypothesis","embInver_rate","embExpStop_rate","embInver_success_words","embExpStop_success_words","mask_rate","mask_rate_ExpStop","mask_success_words","mask_Expstop_success_words"]].to_csv(csv_output_path, index=False, encoding='utf-8')

    elif args.dataset == "piidocs":
        df.rename(columns={'label': 'ground_truth', 'text': 'text', 'new_content': 'perturbed_sentence','EmbInver_success_rate':'embInver_rate',
        "EmbExpStop_success_rate":"embExpStop_rate",
        "EmbInver_success_words":"embInver_success_words",
        "EmbExpStop_success_words":"embExpStop_success_words",
        "Mask_success_rate":"mask_rate",
        "Mask_Expstop_success_rate":"mask_rate_ExpStop",
        "Mask_success_words":"mask_success_words",
        "Mask_Expstop_success_words":"mask_Expstop_success_words"
        }, inplace=True)
        perturbed_data = df[['text', 'ground_truth', 'perturbed_sentence',"embInver_rate","embExpStop_rate","embInver_success_words","embExpStop_success_words","mask_rate","mask_rate_ExpStop","mask_success_words","mask_Expstop_success_words"]].to_dict(orient='records')
        df[['text', 'ground_truth', 'perturbed_sentence',"embInver_rate","embExpStop_rate","embInver_success_words","embExpStop_success_words","mask_rate","mask_rate_ExpStop","mask_success_words","mask_Expstop_success_words"]].to_csv(csv_output_path, index=False, encoding='utf-8')

    with open(json_output_path, 'w', encoding='utf-8') as jf:
        json.dump(perturbed_data, jf, ensure_ascii=False, indent=4)

    
    return df