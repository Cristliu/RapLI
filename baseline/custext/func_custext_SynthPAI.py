import os
import json
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


warnings.filterwarnings('ignore')

parser = get_parser()
args = parser.parse_args()

def word_normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

def load_jsonl_data(file_path):
    """
    Load data in jsonl format and extract text that needs to be processed
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if "comments" in item:
                    for comment in item["comments"]:
                        if "text" in comment:
                            # Save original data structure for later updates
                            data.append({
                                "username": item.get("username", ""),
                                "comment_text": comment["text"],
                                "comment_obj": comment,
                                "item_obj": item
                            })
            except json.JSONDecodeError:
                print(f"Cannot parse line: {line.strip()[:100]}...")
                continue
    return pd.DataFrame(data)

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
                ###Here we map all words without loading vocabulary in advance
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
        
        # Ensure directory exists
        if not os.path.exists(f"./NoiseResults/{args.task}/{args.dataset}/Time"):
            os.makedirs(f"./NoiseResults/{args.task}/{args.dataset}/Time", exist_ok=True)
        
        #Save this time to a json file
        time_data = {
            "load_embeddings_time": end_time - start_time
        }
        with open(f"./NoiseResults/{args.task}/{args.dataset}/Time/custext_load_embeddings_time.json", 'w', encoding='utf-8') as time_file:
            json.dump(time_data, time_file, ensure_ascii=False, indent=4)

        # Convert to PyTorch tensor #otherwise running this else branch will cause an error
        embeddings = torch.from_numpy(embeddings_np).to(device)

    return embeddings, word2idx, idx2word

def get_customized_mapping_synthpai(eps, top_k):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings, word2idx, idx2word = load_embeddings(device)
    
    # Load SynthPAI data
    df_test = load_jsonl_data(f"./data/synthpai/comments_eval_revised.jsonl")
    
    # Create corpus for SynthPAI data
    test_corpus = " ".join(df_test.comment_text)
    
    word_freq = [x[0] for x in Counter(test_corpus.split()).most_common()]
    print(f"Total number of unique words in the SynthPAI corpus: {len(word_freq)}")
    
    sim_word_dict = defaultdict(list)
    p_dict = defaultdict(list)
    
    for i in trange(len(word_freq)):
        word = word_freq[i]
        if word in word2idx:
            word_idx = word2idx[word]
            word_embedding = embeddings[word_idx].to(device)  # Shape: [1, embedding_dim]
            
            distances = torch.norm(embeddings - word_embedding, dim=1)
            
            # Get indices of the top_k words with closest distances
            sorted_distances, indices = torch.topk(distances, top_k, largest=False)
            word_list = [idx2word[idx] for idx in indices.cpu().numpy()]
            
            # Calculate probability distribution
            min_dist = sorted_distances.min()
            max_dist = sorted_distances.max()
            range_dist = max_dist - min_dist
            if range_dist == 0:
                range_dist = 1e-6  # Avoid division by zero
            #Normalize distances    
            new_sim_dist_list = - (sorted_distances - min_dist) / range_dist
            tmp = torch.exp(eps * new_sim_dist_list / 2)
            p = tmp / tmp.sum()
            p_dict[word] = p.cpu().numpy().tolist()

            sim_word_dict[word] = word_list

    # Ensure directory exists
    os.makedirs(f"./data/custext_dict", exist_ok=True)
    
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

    return sim_word_dict, p_dict

def save_disturbed_jsonl(data, output_path):
    """
    Save the perturbed text back to the original jsonl format
    """
    # First group by username to reconstruct the original data structure
    grouped_data = {}
    for _, row in data.iterrows():
        username = row["username"]
        if username not in grouped_data:
            grouped_data[username] = row["item_obj"]
            
        # Update the corresponding text in the comments section of the item
        for comment in grouped_data[username]["comments"]:
            if comment is row["comment_obj"]:
                comment["text"] = row["perturbed_sentence"]

    # Write to new jsonl file
    with open(output_path, 'w', encoding='utf-8') as f:
        for username, item_obj in grouped_data.items():
            f.write(json.dumps(item_obj, ensure_ascii=False) + '\n')

def generate_new_sents_synthpai(df, sim_word_dict, p_dict, save_stop_words):
    """
    Generate perturbed text for SynthPAI data
    """
    stop_words = set(stopwords.words('english'))

    cnt = 0
    raw_cnt = 0
    stop_cnt = 0
    dataset = df.comment_text
    new_dataset = []

    for i in trange(len(dataset)):
        record = dataset[i].split()
        new_record = []
        for word in record:
            if (save_stop_words and word in stop_words) or (word not in sim_word_dict):
                if word in stop_words:
                    stop_cnt += 1
                    raw_cnt += 1
                if is_number(word):
                    try:
                        word = str(round(float(word)) + np.random.randint(1000))
                    except:
                        pass
                new_record.append(word)
            else:
                p = np.array(p_dict[word])
                if np.isnan(p).any():
                    print(f"Skipping word '{word}' due to NaN in probabilities")
                    new_record.append(word)
                    continue
                # Normalize probability array
                p = p / p.sum()
                new_word = np.random.choice(sim_word_dict[word], p=p)
                new_record.append(new_word)
                if new_word == word:
                    raw_cnt += 1
            cnt += 1
        
        new_dataset.append(" ".join(new_record))

    df['perturbed_sentence'] = new_dataset

    # Ensure output directory exists
    output_dir = f"./NoiseResults/{args.task}/{args.dataset}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Output file paths
    json_output_path = f"{output_dir}/custext_epsilon_{args.epsilon}.json"
    csv_output_path = f"{output_dir}/custext_epsilon_{args.epsilon}.csv"
    jsonl_output_path = f"{output_dir}/custext_epsilon_{args.epsilon}_disturbed.jsonl"

    save_disturbed_jsonl(df, jsonl_output_path)
    print(f"Perturbed JSONL file has been saved to: {jsonl_output_path}")
    
    perturbed_data = df[['comment_text', 'perturbed_sentence']].rename(
        columns={'comment_text': 'text'}
    ).to_dict(orient='records')
    
    with open(json_output_path, 'w', encoding='utf-8') as jf:
        json.dump(perturbed_data, jf, ensure_ascii=False, indent=4)
    
    df[['comment_text', 'perturbed_sentence']].rename(
        columns={'comment_text': 'text'}
    ).to_csv(csv_output_path, index=False, encoding='utf-8')

    return df