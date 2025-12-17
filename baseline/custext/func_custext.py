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


warnings.filterwarnings('ignore')

parser = get_parser()
args = parser.parse_args()

def word_normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

def load_data(dataset=None):
    print(f'__loading__{args.dataset}__')
    if args.dataset == "ag_news":
        test_df = pd.read_csv(f"./data/{args.task}/{args.dataset}/test.tsv", sep='\t')
    elif args.dataset == "piidocs":
        test_df = pd.read_csv(f"./data/{args.task}/{args.dataset}/piidocs.tsv", sep='\t')

    # Added support for SAMSUM dataset
    elif args.dataset == "samsum":
        # Read JSON file
        with open(f"./data/{args.task}/{args.dataset}/samsum_combined.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert JSON data to DataFrame
        test_df = pd.DataFrame({
            'dialogue': [item['dialogue'] for item in data],
            'summary': [item['summary'] for item in data]
        })


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
        #Save this time to a json file
        time_data = {
            "load_embeddings_time": end_time - start_time
        }
        with open(f"./NoiseResults/{args.task}/{args.dataset}/Time/custext_load_embeddings_time.json", 'w', encoding='utf-8') as time_file:
            json.dump(time_data, time_file, ensure_ascii=False, indent=4)

        # Convert to PyTorch tensor #otherwise running this else branch will cause an error
        embeddings = torch.from_numpy(embeddings_np).to(device)

    return embeddings, word2idx, idx2word

def get_customized_mapping(eps, top_k):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings, word2idx, idx2word = load_embeddings(device)
    df_test = load_data(args.dataset)
    
    if args.dataset == "ag_news":
        test_corpus = " ".join(df_test.text)
    elif args.dataset == "piidocs":
        test_corpus = " ".join(df_test.text)
    # Add SAMSUM corpus construction
    elif args.dataset == "samsum":
        test_corpus = " ".join(df_test.dialogue)
    
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
            
            # Get indices of the top_k words with closest distances
            sorted_distances, indices = torch.topk(distances, top_k, largest=False)
            word_list = [idx2word[idx] for idx in indices.cpu().numpy()]
            # embedding_list = embeddings[indices]
            
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

def generate_new_sents_s1(df, sim_word_dict, p_dict, save_stop_words, type="test"):

    stop_words = set(stopwords.words('english'))

    cnt = 0
    raw_cnt = 0
    stop_cnt = 0
    if args.dataset == "ag_news":
        dataset = df.text
    elif args.dataset == "piidocs":
        dataset = df.text
    # Add SAMSUM dataset processing
    elif args.dataset == "samsum":
        dataset = df.dialogue

    new_dataset = []

    for i in trange(len(dataset)):
        record = dataset[i].split()
        new_record = []
        for word in record:
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
        
        # return 0
        
        new_dataset.append(" ".join(new_record))

    df['new_content'] = new_dataset

    output_dir = f"./NoiseResults/{args.task}/{args.dataset}" #custext_epsilon_{args.epsilon}

    if type == "test":
        json_output_path = f"{output_dir}/custext_epsilon_{args.epsilon}.json"
        csv_output_path = f"{output_dir}/custext_epsilon_{args.epsilon}.csv"

    if args.dataset == "ag_news":
        df.rename(columns={'label': 'ground_truth', 'text': 'text', 'new_content': 'perturbed_sentence'}, inplace=True)
        perturbed_data = df[['text', 'ground_truth', 'perturbed_sentence']].to_dict(orient='records')
        df[['text', 'ground_truth', 'perturbed_sentence']].to_csv(csv_output_path, index=False, encoding='utf-8')

    elif args.dataset == "piidocs":
        df.rename(columns={'label': 'ground_truth', 'text': 'text', 'new_content': 'perturbed_sentence'}, inplace=True)
        perturbed_data = df[['text', 'ground_truth', 'perturbed_sentence']].to_dict(orient='records')
        df[['text', 'ground_truth', 'perturbed_sentence']].to_csv(csv_output_path, index=False, encoding='utf-8')

    # Add processing for SAMSUM dataset
    elif args.dataset == "samsum":
        df.rename(columns={'dialogue': 'text', 'summary': 'ground_truth', 'new_content': 'perturbed_sentence'}, inplace=True)
        perturbed_data = df[['text', 'ground_truth', 'perturbed_sentence']].to_dict(orient='records')
        df[['text', 'ground_truth', 'perturbed_sentence']].to_csv(csv_output_path, index=False, encoding='utf-8')




    with open(json_output_path, 'w', encoding='utf-8') as jf:
        json.dump(perturbed_data, jf, ensure_ascii=False, indent=4)

    return df