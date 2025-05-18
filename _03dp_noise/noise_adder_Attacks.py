import torch
import random
import re
import numpy as np
from nltk.corpus import stopwords
import torch.nn.functional as F

class DPNoiseAdder:
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_matrix = model.get_input_embeddings().weight.to(self.device)  # (vocab_size, embedding_dim)

        # Get the list of special tokens
        self.special_token_ids = tokenizer.all_special_ids
        self.special_tokens = tokenizer.all_special_tokens

        # Get stop words
        self.stop_words = set(stopwords.words('english'))

        # Pre-filter the vocabulary, keeping only English words
        self.filtered_vocab = self.filter_vocab()
        print(f"Filtered vocab size: {len(self.filtered_vocab)}")

        self.args = args
    
    def filter_vocab(self):  # Same as non-attack case
        filtered_vocab = []
        for idx in range(self.embedding_matrix.size(0)):
            token = self.tokenizer.decode([idx])
            token_normalized = token.strip()
            # Check if the token contains non-ASCII characters
            if all(ord(c) < 128 for c in token_normalized):
                # Check if it contains letters or numbers, or if it is enclosed in []
                if not re.search(r'[a-zA-Z0-9]', token_normalized) or re.match(r'^\[.*\]$', token_normalized):
                    # print(f"Filtered token: {token_normalized}")
                    continue
                
                filtered_vocab.append(idx)
        return filtered_vocab
    

    def calculate_k(self, epsilon, is_high_risk):  # Ablation experiment 3: Different values for K_base, same as non-attack case
        if self.args.ablation_3_1:
            K_base = 10
        elif self.args.ablation_3_2:
            K_base = 50
        elif self.args.ablation_3_3:
            K_base = 100
        else:
            K_base = 20
        
        if is_high_risk:
            return K_base 
        return int(K_base + round(100 / (epsilon ** 1.2)))
    
    def replace_number(self, token):  # Same as non-attack case
        # Check if it is an integer or decimal
        if re.match(r'^\d+(\.\d+)?$', token):
            original_num = float(token)
            # Generate a completely random number
            new_num = original_num
            while new_num == original_num:
                new_num = random.randint(1, 1000)
            return str(new_num)  # No space added
                
        # Check if it is an ordinal number (e.g., 1st, 2nd, 3rd, 4th, etc.)
        elif re.match(r'^\d+(st|nd|rd|th)$', token):
            # Generate a new random ordinal number
            new_num = random.randint(1, 100)
            # Determine the ordinal suffix
            if 10 <= new_num % 100 <= 20:
                suffix = 'th'
            else:
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(new_num % 10, 'th')
            return f"{new_num}{suffix}"  # No space added
        else:
            return token

    def replace_currency_symbol(self, token):  # Same as non-attack case
        # Replace currency symbols
        currency_symbols = ['$', '€', '£', '¥', '₹', '₩', '₽', '₫', '฿', '₴', '₦']
        pattern = r'[' + re.escape(''.join(currency_symbols)) + r']'
        replacement_symbol = random.choice([sym for sym in currency_symbols if sym not in token])
        new_token = re.sub(pattern, replacement_symbol, token)

        return new_token



    def get_replacement(self, token, is_high_risk, epsilon_effective):  # Same as non-attack case
        """
        For a given token, generate a list of candidate replacement words and corresponding selection probabilities
        """
        # Get the ID of the token
        # If the ID is not found, it should be the unk_token_id
        target_token_id = self.tokenizer.convert_tokens_to_ids(token)
        # print(f"token: {token}, target_token_id: {target_token_id}")
        unk_token_id = self.tokenizer.unk_token_id
        # print(f"unk_token_id: {unk_token_id}")
        
        # Decode the token and preprocess it
        decoded_token = self.tokenizer.decode([target_token_id])
        decoded_token_lower = re.sub(r'\s+', '', decoded_token.lower())


        # Handle numbers or ordinal numbers
        if re.match(r'^\d+(\.\d+)?$', decoded_token_lower) or re.match(r'^\d+(st|nd|rd|th)$', decoded_token_lower):
            # Return a randomly generated number or ordinal number as a replacement
            new_token = self.replace_number(decoded_token_lower)
            return [new_token], [1.0]

        # Handle currency symbols
        if decoded_token_lower in ['$', '€', '£', '¥', '₹', '₩', '₽', '₫', '฿', '₴', '₦']:
            new_token = self.replace_currency_symbol(decoded_token_lower)
            return [new_token], [1.0]


        # Exclude special tokens
        if decoded_token_lower in self.special_tokens:
            return [token], [1.0]

        # Retain words that do not need to be replaced and non-alphabetic tokens
        if decoded_token_lower.lower() in self.stop_words or not decoded_token_lower.isalpha():
            return [token], [1.0]

        # Only get embeddings for retained words
        # If the token ID is not equal to unk_token_id, directly use the embedding corresponding to that ID
        if target_token_id != unk_token_id:
            token_embedding = self.embedding_matrix[target_token_id].unsqueeze(0)

        else:
            inputs = self.tokenizer(token, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            token_embedding = outputs.last_hidden_state[:, 1, :].detach().unsqueeze(0)

        # Get the nearest neighbors
        sim_words, sim_scores_normalized = self.get_top_k_similar_words(
            token_embedding, epsilon_effective, is_high_risk
        )

        # Ablation experiment 4: Do not perturb is_high_risk tokens
        if not self.args.ablation_5:
            if is_high_risk:
            # Processing method: Reverse the probability matrix, i.e., swap the first element with the last element, the second element with the second-to-last element, and so on
                sim_scores_normalized = sim_scores_normalized[::-1]

        # Use the exponential mechanism to calculate selection probabilities
        delta_u = 1  # Sensitivity of similarity scores
        exponents = np.exp((epsilon_effective * sim_scores_normalized) / (2 * delta_u))
        # print(f"exponents: {exponents}")
        probabilities = exponents / np.sum(exponents)
        return sim_words, probabilities

    def EmbInver_attack(self, token, EmbInver_K):
        target_token_id = self.tokenizer.convert_tokens_to_ids(token)
        unk_token_id = self.tokenizer.unk_token_id
        if target_token_id != unk_token_id:
            token_embedding = self.embedding_matrix[target_token_id].unsqueeze(0)

        else:
            inputs = self.tokenizer(token, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            token_embedding = outputs.last_hidden_state[:, 1, :].detach().unsqueeze(0)

        similar_words_K = self.Direct_get_top_k_similar_words_for_attack(token_embedding, EmbInver_K)
        return similar_words_K

    def Direct_get_top_k_similar_words_for_attack(self, token_embedding, EmbInver_K):
        K = EmbInver_K
        cosine_similarities = F.cosine_similarity(token_embedding, self.embedding_matrix[self.filtered_vocab], dim=1)
        top_values, top_indices = torch.topk(cosine_similarities, k=K)
        if top_indices.dim() > 1:
            top_indices = top_indices.squeeze()
        if top_indices.dim() == 0:
            top_indices = top_indices.unsqueeze(0)
        similar_words_K = [self.tokenizer.decode([self.filtered_vocab[idx]]) for idx in top_indices.cpu().numpy()]

        return similar_words_K

    def get_top_k_similar_words(self, token_embedding, epsilon, is_high_risk):  # Same as non-attack case
        K = self.calculate_k(epsilon,is_high_risk)
        similar_words = []
        # similar_words_normalized = []  # Store normalized words to avoid selecting duplicate words
        similar_scores = []
    
        cosine_similarities = F.cosine_similarity(token_embedding, self.embedding_matrix[self.filtered_vocab], dim=1)
        # Directly select the top K words with the highest cosine similarities
        top_values, similar_indices = torch.topk(cosine_similarities, k=K)
        
        similar_words = [self.tokenizer.decode([self.filtered_vocab[idx]]) for idx in similar_indices.cpu().numpy()]
        similar_scores = top_values.detach().cpu().numpy()

        # Normalize similarity scores
        sim_scores = np.array(similar_scores)
        u_max = np.max(sim_scores)
        u_min = np.min(sim_scores)
        if u_max - u_min > 0:
            sim_scores_normalized = (sim_scores - u_min) / (u_max - u_min)
        else:
            print("Warning: Similarity scores are not normalized.")

        return similar_words, sim_scores_normalized

    def select_replacement(self, sim_words, probabilities):  # Same as non-attack case
        new_token = np.random.choice(sim_words, p=probabilities)
        return new_token