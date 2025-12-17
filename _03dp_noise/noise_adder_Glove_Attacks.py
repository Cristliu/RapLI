import os
import string
import torch
import torch.nn.functional as F
import numpy as np
import random
import re
from nltk.corpus import stopwords
import tqdm
from transformers import AutoTokenizer, AutoModel


class DPNoiseAdder:
    """
    Use GloVe pre-trained vectors to replace tokens.
    Tokens that cannot be found are retained directly.
    """
    def __init__(self, args, glove_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.args = args
        self.embedding_matrix, self.word2id, self.id2word = self.load_glove_embeddings(glove_path)

        # Stop words, special tokens, etc. can be added as needed
        self.stop_words = set(stopwords.words('english'))
        self.punct = set(string.punctuation)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = self.load_model_and_tokenizer("bert-base-uncased")

    def load_model_and_tokenizer(self, model_name):
        """
        Load tokenizer and model based on the given model name, and move to GPU (if available)
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(self.device)
        return model, tokenizer

    def load_glove_embeddings(self, glove_path):
        """
        Load GloVe vectors from glove_path into self.glove_dict
        """
        embedding_dir = os.path.dirname(glove_path)
        npy_path = os.path.join(embedding_dir, f'santext_{self.args.dataset}_glove_840B-300d.npy')
        word2idx_path = os.path.join(embedding_dir, f"santext_{self.args.dataset}_glove_840B-300d_word2idx.npy")

        if os.path.exists(npy_path) and os.path.exists(word2idx_path):
            print("Loading GloVe embeddings from cached files...")
            embeddings = np.load(npy_path)
            word2id = np.load(word2idx_path, allow_pickle=True).item()
            # Define id2word as a list
            id2word = [None] * len(word2id)
            for word, idx in word2id.items():
                if idx < len(id2word):
                    id2word[idx] = word
                else:
                    print(f"Index {idx} for word '{word}' is out of range.")
            # Check for None positions
            for idx, word in enumerate(id2word):
                if word is None:
                    print(f"Warning: No word found for index {idx}.")
            embeddings = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            return embeddings, word2id, id2word
        
        print("Loading GloVe embeddings from text file...")
        embeddings = []
        word2id = {}
        id2word = []
        all_count = 0

        num_lines = sum(1 for _ in open(glove_path, 'r', encoding='utf-8'))
        with open(glove_path, 'r', encoding='utf-8') as f:
            for row in tqdm(f, total=num_lines, desc="Loading embeddings"):
                values = row.rstrip().split(' ')
                word = values[0]
                try:
                    vector = np.asarray(values[1:], dtype='float32')
                    if len(vector) != 300:
                        continue
                except ValueError:
                    continue
                if word.isalpha() and word.lower() not in self.stop_words and word not in self.punct:
                    if word not in word2id:
                        word2id[word] = all_count
                        id2word.append(word)  # Use list, index corresponds to embeddings
                        embeddings.append(vector)
                        all_count += 1

        embeddings = np.array(embeddings, dtype='float32')
        np.save(npy_path, embeddings)
        np.save(word2idx_path, word2id)

        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        return embeddings, word2id, id2word

    def calculate_k(self, epsilon, is_high_risk):
        K_base = 20
        if is_high_risk:
            return K_base 
        return int(K_base + round(100 / (epsilon ** 1.2)))
    
    def replace_number(self, token):
        if re.match(r'^\d+(\.\d+)?$', token):
            original_num = float(token)
            new_num = original_num
            while new_num == original_num:
                new_num = random.randint(1, 1000)
            return str(new_num)
                
        elif re.match(r'^\d+(st|nd|rd|th)$', token):
            new_num = random.randint(1, 100)
            if 10 <= new_num % 100 <= 20:
                suffix = 'th'
            else:
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(new_num % 10, 'th')
            return f"{new_num}{suffix}"
        else:
            return token

    def replace_currency_symbol(self, token):
        currency_symbols = ['$', '€', '£', '¥', '₹', '₩', '₽', '₫', '฿', '₴', '₦']
        pattern = r'[' + re.escape(''.join(currency_symbols)) + r']'
        replacement_symbol = random.choice([sym for sym in currency_symbols if sym not in token])
        new_token = re.sub(pattern, replacement_symbol, token)
        return new_token

    def get_glove_replacement(self, token, is_high_risk, epsilon_effective):
        if re.match(r'^\d+(\.\d+)?$', token) or re.match(r'^\d+(st|nd|rd|th)$', token):
            new_token = self.replace_number(token)
            return [new_token], [1.0]

        if token in ['$', '€', '£', '¥', '₹', '₩', '₽', '₫', '฿', '₴', '₦']:
            new_token = self.replace_currency_symbol(token)
            return [new_token], [1.0]

        if token.lower() in self.stop_words or not token.isalpha():
            return [token], [1.0]

        embeddings = self.embedding_matrix
        word2id = self.word2id
        id2word = self.id2word
        
        if token in word2id:
            idx = word2id[token]
            word_vector = embeddings[idx].unsqueeze(0)

            cosine_similarities = F.cosine_similarity(word_vector, embeddings, dim=1)
            similar_indices = torch.argsort(cosine_similarities, descending=True).cpu().numpy()
            K = self.calculate_k(epsilon_effective, is_high_risk)
            top_similarities = cosine_similarities[similar_indices[:K]].cpu().numpy()
            top_indices = similar_indices[:K]

            min_cos = top_similarities.min()
            max_cos = top_similarities.max()
            min_max_cos = max_cos - min_cos
            if min_max_cos == 0:
                new_sim_Cos_list = np.zeros_like(top_similarities)
            else:
                new_sim_Cos_list = (top_similarities - min_cos) / min_max_cos

            if is_high_risk:
                new_sim_Cos_list = new_sim_Cos_list[::-1]
            
            tmp = np.exp(epsilon_effective * new_sim_Cos_list / 2)
            norm = tmp.sum()
            if norm == 0:
                p = np.full_like(tmp, 1.0 / len(tmp))
            else:
                p = tmp / norm

            sim_words = []
            valid_p = []
            for i, prob in zip(top_indices, p):
                if i < len(id2word):
                    word = id2word[i]
                    if word:
                        sim_words.append(word)
                        valid_p.append(prob)
                    else:
                        print(f"Warning: No word found for index {i}.")
                else:
                    print(f"Warning: Index {i} is out of range.")
                if len(sim_words) == K:
                    break

            if len(sim_words) == K:
                sim_word_dict = sim_words
                p_dict = valid_p
            else:
                print(f"Warning: Only {len(sim_words)} similar words found for '{token}'.")
                while len(sim_words) < K:
                    sim_words.append(token)
                    valid_p.append(0.0)
                valid_p = np.array(valid_p)
                if valid_p.sum() != 0:
                    valid_p /= valid_p.sum()
                else:
                    valid_p = np.full_like(valid_p, 1.0 / len(valid_p))
                sim_word_dict = sim_words
                p_dict = valid_p.tolist()
                p = np.array(p_dict)
                p = p / p.sum() if p.sum() != 0 else np.full_like(p, 1.0 / len(p))

            return sim_word_dict, p

        else:
            return [token], [1.0]

    def EmbInver_attack(self, token, EmbInver_K):
        if token in self.word2id:
            idx = self.word2id[token]
            token_embedding = self.embedding_matrix[idx].unsqueeze(0)
        else:
            return [token]

        similar_words_K = self.Direct_get_top_k_similar_words_for_attack(token_embedding, EmbInver_K)
        return similar_words_K

    def Direct_get_top_k_similar_words_for_attack(self, token_embedding, EmbInver_K):
        cosine_similarities = F.cosine_similarity(token_embedding, self.embedding_matrix, dim=1)
        top_indices = torch.topk(cosine_similarities, k=EmbInver_K).indices
        similar_words_K = [self.id2word[idx] for idx in top_indices.cpu().numpy()]
        return similar_words_K

    def select_replacement(self, token, sim_words, probabilities):
        if any(np.isnan(probabilities)):
            print(f"Skipping word '{token}' due to NaN in probabilities")
            return token
        try:
            new_token = np.random.choice(sim_words, p=probabilities)
        except ValueError as ve:
            print(f"Error for token '{token}': {ve}")
            return token
        return new_token