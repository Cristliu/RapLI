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
        self.embedding_matrix = model.get_input_embeddings().weight.to(self.device)  

        # Get special tokens list
        self.special_token_ids = tokenizer.all_special_ids
        self.special_tokens = tokenizer.all_special_tokens

        # Get stop words
        self.stop_words = set(stopwords.words('english'))

        # Pre-filter vocabulary, keep only English words
        self.filtered_vocab = self.filter_vocab()
        print(f"Filtered vocab size: {len(self.filtered_vocab)}")

        self.args = args
    
    def filter_vocab(self):
        filtered_vocab = []
        for idx in range(self.embedding_matrix.size(0)):
            token = self.tokenizer.decode([idx])
            token_normalized = token.strip()
            # Check if token contains non-ASCII characters
            if all(ord(c) < 128 for c in token_normalized):
                # Check if contains letters or numbers, or if it's wrapped in []
                if not re.search(r'[a-zA-Z0-9]', token_normalized) or re.match(r'^\[.*\]$', token_normalized):
                    continue
                
                filtered_vocab.append(idx)
        return filtered_vocab


    def calculate_k(self, epsilon, is_high_risk):######Ablation experiment 3: Different K_base values #Consistent with non-attack scenario
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
    
    def replace_number(self, token):#Consistent with non-attack scenario
        # Check if it's an integer or decimal
        if re.match(r'^\d+(\.\d+)?$', token):
            original_num = float(token)
            # Generate a completely random number
            new_num = original_num
            while new_num == original_num:
                new_num = random.randint(1, 1000)
            return str(new_num)  # No space added
                
        # Check if it's an ordinal number (like 1st, 2nd, 3rd, 4th, etc.)
        elif re.match(r'^\d+(st|nd|rd|th)$', token):
            # Generate a new random ordinal
            new_num = random.randint(1, 100)
            # Determine ordinal suffix
            if 10 <= new_num % 100 <= 20:
                suffix = 'th'
            else:
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(new_num % 10, 'th')
            return f"{new_num}{suffix}"  # No space added
        else:
            return token

    def replace_currency_symbol(self, token):#Consistent with non-attack scenario
        # Replace currency symbols
        currency_symbols = ['$', '€', '£', '¥', '₹', '₩', '₽', '₫', '฿', '₴', '₦']
        pattern = r'[' + re.escape(''.join(currency_symbols)) + r']'
        replacement_symbol = random.choice([sym for sym in currency_symbols if sym not in token])
        new_token = re.sub(pattern, replacement_symbol, token)

        return new_token


    def get_replacement(self, token, is_high_risk, epsilon_effective):#Consistent with non-attack scenario
        """
        Generate a list of candidate replacement words and corresponding selection probabilities for a given token
        """
        # Get token ID---when not found, it should be unk_token_id
        target_token_id = self.tokenizer.convert_tokens_to_ids(token)
        # print(f"token: {token}, target_token_id: {target_token_id}")
        unk_token_id = self.tokenizer.unk_token_id
        # print(f"unk_token_id: {unk_token_id}")
        
        # Decode token and preprocess
        decoded_token = self.tokenizer.decode([target_token_id])
        # print(f"decoded_token1: {decoded_token}")
        decoded_token_lower = re.sub(r'\s+', '', decoded_token.lower())
        # print(f"decoded_token2: {decoded_token}") 


        # Handle numbers or ordinals
        if re.match(r'^\d+(\.\d+)?$', decoded_token_lower) or re.match(r'^\d+(st|nd|rd|th)$', decoded_token_lower):
            # Return randomly generated number or ordinal as replacement
            new_token = self.replace_number(decoded_token_lower)
            return [new_token], [1.0]

        # Handle currency symbols
        if decoded_token_lower in ['$', '€', '£', '¥', '₹', '₩', '₽', '₫', '฿', '₴', '₦']:
            new_token = self.replace_currency_symbol(decoded_token_lower)
            return [new_token], [1.0]


        # Exclude special tokens
        if decoded_token_lower in self.special_tokens:
            return [token], [1.0]

        # Preserve tokens that don't need replacement and non-alphabetic tokens
        if decoded_token_lower.lower() in self.stop_words or not decoded_token_lower.isalpha():
            return [token], [1.0]

###Only words that remain need to get embeddings
        # If token ID is not equal to unk_token_id, directly use the embedding representation of that ID
        if target_token_id != unk_token_id:
            token_embedding = self.embedding_matrix[target_token_id].unsqueeze(0)

        else:
        # Ablation experiment 4: Here if not found, keep this token directly, that is, return sim_words, probabilities as token, 1.0
            if self.args.ablation_4:
                return [token], [1.0]
            else:
            ###Processing method: If not found, input to model to get embedding
                inputs = self.tokenizer(token, return_tensors='pt').to(self.device)
                outputs = self.model(**inputs)
                token_embedding = outputs.last_hidden_state[:, 1, :].detach().unsqueeze(0)

        # Get neighboring table
        sim_words, sim_scores_normalized = self.get_top_k_similar_words(
            token_embedding, epsilon_effective, is_high_risk
        )

        ###Ablation experiment 5: No perturbation for is_high_risk tokens
        if not self.args.ablation_5:
            if is_high_risk:
            #Processing method: Perform a reversal operation on the probability matrix, i.e., swap the first element with the last element, the second element with the second-to-last element... and so on
                sim_scores_normalized = sim_scores_normalized[::-1]

        # Use exponential mechanism to calculate selection probabilities
        delta_u = 1  # Sensitivity of similarity scores
        exponents = np.exp((epsilon_effective * sim_scores_normalized) / (2 * delta_u))
        # print(f"exponents: {exponents}")
        probabilities = exponents / np.sum(exponents)####Normalize after exp, CUSTEXT also has this step===the formula also has this step, i.e., divide by the sum of exp////but SANTEXT code doesn't seem to have this step but directly uses softmax (equivalent)===》Softmax function converts input vectors to probability distributions. Each output value represents the probability of the corresponding category, and the sum of all output values is 1.

        return sim_words, probabilities

    def EmbInver_attack(self, token, EmbInver_K):
        """
        Embedding Inversion Attack - attempt to infer the original token from the perturbed token
        """
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
        """
        Directly get the K most similar words
        """
        K = EmbInver_K
        cosine_similarities = F.cosine_similarity(token_embedding, self.embedding_matrix[self.filtered_vocab], dim=1)
        top_values, top_indices = torch.topk(cosine_similarities, k=K)
        if top_indices.dim() > 1:
            top_indices = top_indices.squeeze()
        if top_indices.dim() == 0:
            top_indices = top_indices.unsqueeze(0)
        similar_words_K = [self.tokenizer.decode([self.filtered_vocab[idx]]) for idx in top_indices.cpu().numpy()]

        return similar_words_K

    def get_top_k_similar_words(self, token_embedding, epsilon, is_high_risk):#Consistent with non-attack scenario
        K = self.calculate_k(epsilon,is_high_risk)
        similar_words = []
        # similar_words_normalized = []  # Store normalized words to avoid selecting duplicate words
        similar_scores = []

        ### Ablation experiment 6: Use Euclidean distance when calculating distance
        if self.args.ablation_6:
            euclidean_distances = torch.norm(token_embedding - self.embedding_matrix[self.filtered_vocab], p=2, dim=1)
            top_values, similar_indices = torch.topk(-euclidean_distances, k=K)
        else:            
            cosine_similarities = F.cosine_similarity(token_embedding, self.embedding_matrix[self.filtered_vocab], dim=1)#dim=1 specifies in each row (i.e., the column dimension of the word vector itself), which is OK
            #Modified to directly select the K words with the highest cosine_similarities
            top_values, similar_indices = torch.topk(cosine_similarities, k=K)
        
        similar_words = [self.tokenizer.decode([self.filtered_vocab[idx]]) for idx in similar_indices.cpu().numpy()]
        similar_scores = top_values.detach().cpu().numpy()

        # Normalize similarity scores
        sim_scores = np.array(similar_scores)
        u_max = np.max(sim_scores)
        u_min = np.min(sim_scores)
        if u_max - u_min > 0:
            if self.args.ablation_6:###Ablation experiment 6: Take negative for Euclidean distance, minimum distance is maximum similarity
                sim_scores_normalized = - (sim_scores - u_min) / (u_max - u_min)
            else:
                sim_scores_normalized = (sim_scores - u_min) / (u_max - u_min)
        else:
            print("Warning: Similarity scores are not normalized.")

        return similar_words, sim_scores_normalized


    def select_replacement(self, sim_words, probabilities):
        """
        Select replacement word
        """
        new_token = np.random.choice(sim_words, p=probabilities)
        return new_token