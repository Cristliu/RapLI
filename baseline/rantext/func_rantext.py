import json
import tiktoken
import random
import tqdm
from decimal import getcontext
import numpy as np
import torch
from transformers import GPT2Tokenizer
import os

getcontext().prec = 100  # Set the precision for decimal calculations to 100 digits to ensure high-precision calculation, especially when handling floating-point numbers.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Initialize only once
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def calculate_distance_matrix_gpu(vector_matrix):
    """
    Use torch.cdist to calculate the distance matrix at once, reducing Python loop overhead.
    """
    # Calculate Euclidean distance directly on GPU
    dist_mat = torch.cdist(vector_matrix, vector_matrix, p=2)
    return dist_mat

def get_tokens(text):
    tokens = tokenizer.tokenize(text)
    tokenized_string = tokenizer.convert_tokens_to_string(tokens)
    return tokenized_string

def add_laplace_noise_to_vector(vector, epsilon, delta_f_new=None):
    vector = torch.tensor(vector, dtype=torch.float64, device=device)
    # Ensure delta_f_new is a PyTorch tensor
    if delta_f_new is not None and not isinstance(delta_f_new, torch.Tensor):
        delta_f_new = torch.tensor(delta_f_new, dtype=torch.float64, device=device)

    if not os.path.exists(f'./data/rantext_json/sorted_cl100_embeddings.json'):
        with open("./data/rantext_json/cl100_embeddings.json", 'r') as f:
            data_t = json.load(f)
            data = {k: data_t[k] for k in list(data_t.keys())}
            data_t = None
        word_list = list(data.keys())
        vector_matrix = torch.tensor(list(data.values()), dtype=torch.float64, device=device)
        data = None
        n_vectors = len(word_list)
        # Use cdist once to calculate all distances, replacing pair-wise traversal
        distance_matrix = calculate_distance_matrix_gpu(vector_matrix)
        temp_distance_matrix = distance_matrix.cpu().numpy()
        temp_distance_dict_matrix = {}
        for i, word1 in enumerate(word_list):
            for j, word2 in enumerate(word_list):
                pair = tuple(sorted([word1, word2]))
                if pair in temp_distance_dict_matrix:
                    continue
                temp_distance_dict_matrix[str(pair)] = float(temp_distance_matrix[i, j])
        with open('./data/rantext_json/temp_distance_json_path.json', 'w') as f:
            json.dump(temp_distance_dict_matrix, f)
    if os.path.exists(f'./data/rantext_json/temp_distance_json_path.json'):
        with open('./data/rantext_json/temp_distance_json_path.json', 'r') as f:
            temp_distance_dict_matrix = json.load(f)
        word_to_index = {word: idx for idx, word in enumerate(word_list)}
        n = len(word_list)
        temp_distance_matrix = torch.zeros((n, n), device=device)
        for key, value in temp_distance_dict_matrix.items():
            word1, word2 = tuple(key.strip("()").split(", "))
            i = word_to_index[word1.strip("'")]
            j = word_to_index[word2.strip("'")]
            temp_distance_matrix[i, j] = value
            temp_distance_matrix[j, i] = value
        sorted_distance_dict_matrix = {}
        for i, word in enumerate(word_list):
            sorted_indices = torch.argsort(temp_distance_matrix[i])
            sorted_words = [(word_list[j], temp_distance_matrix[i, j].item()) for j in sorted_indices]
            sorted_distance_dict_matrix[word] = sorted_words
        with open('./data/rantext_json/sorted_cl100_embeddings.json', 'w') as f:
            json.dump(sorted_distance_dict_matrix, f)
    if not os.path.exists(f'./data/rantext_json/sensitivity_of_embeddings.json'):
        json_path = "./data/rantext_json/cl100_embeddings.json"
        with open(json_path, 'r') as f:
            vector_data_json = json.load(f)
        word_list = list(vector_data_json.keys())
        vector_matrix = torch.tensor(list(vector_data_json.values()), dtype=torch.float64, device=device)
        n_dimensions = vector_matrix.shape[1]
        delta_f_new = torch.zeros(n_dimensions, device=device)
        for dim in tqdm.trange(n_dimensions):
            dim_data = vector_matrix[:, dim]
            sorted_dim_data = torch.sort(dim_data).values
            differences = sorted_dim_data[-1] - sorted_dim_data[0]
            delta_f_new[dim] = differences
        delta_f_new_json_path = './data/rantext_json/sensitivity_of_embeddings.json'
        with open(delta_f_new_json_path, 'w') as f:
            json.dump(delta_f_new.cpu().numpy().tolist(), f)
    else:
        if delta_f_new is None:
            with open('./data/rantext_json/sensitivity_of_embeddings.json', 'r') as f:
                delta_f_new = torch.tensor(json.load(f), dtype=torch.float64, device=device)
    tt = 0
    # Can use either math.log or convert value to torch.Tensor, demonstrating math.log approach below
    val = epsilon * 19.064721649556482 - 38.1294334077209
    if val > 0:
        # For GPU tensor calculation, can use:
        val_t = torch.tensor(val, dtype=torch.float64, device=device)
        tt = 0.01658160142016071 * torch.log(val_t) + 9.311083811697406
        #
        # Or use math.log (pure CPU calculation):
        # tt = 0.01658160142016071 * math.log(val) + 9.311083811697406
    
    if epsilon < 2:
        beta_values = delta_f_new / epsilon
    else:
        beta_values = delta_f_new / tt
    
    # Ensure beta_values is a PyTorch tensor
    if not isinstance(beta_values, torch.Tensor):
        beta_values = torch.tensor(beta_values, dtype=torch.float64, device=device)
    beta_values = beta_values.to(device)

    noisy_vector = torch.zeros_like(vector, dtype=torch.float64, device=device)
    noise = torch.tensor(np.random.laplace(0, beta_values.cpu().numpy()), dtype=torch.float64, device=device)
    noisy_vector = vector + noise
    return noisy_vector.cpu().numpy()

def perturb_sentence(sent, epsilon, model, token_to_vector_dict, sorted_distance_data, delta_f_new):
    # Ensure delta_f_new is a PyTorch tensor
    if not isinstance(delta_f_new, torch.Tensor):
        delta_f_new = torch.tensor(delta_f_new, dtype=torch.float64, device=device)
    enc = tiktoken.encoding_for_model(model)
    tokens_b = enc.encode(sent)
    tokens = [(enc.decode_single_token_bytes(t)).decode('Latin-1') for t in tokens_b]
    new_tokens = []
    Delta_u = 1.0
    exp_factor = epsilon / (2 * Delta_u)
    for origin_token in tokens:
        if origin_token.isnumeric():
            new_tokens.append(str(random.randint(1, 1000)))
            continue
        if origin_token[0] == ' ':
            origin_token = origin_token[1:]
            ###If there's nothing left after removing spaces, skip
            if not origin_token:
                continue
        origin_embed = token_to_vector_dict.get(origin_token, None)
        if origin_embed is None:
            new_tokens.append(origin_token)  # Keep original token
            continue
        noise_embed = add_laplace_noise_to_vector(origin_embed, epsilon, delta_f_new)
        distance = np.linalg.norm(origin_embed - noise_embed)
        sorted_distances_for_token = sorted_distance_data.get(origin_token, None)
        if sorted_distances_for_token is None:
            new_tokens.append(origin_token)  # Keep original token
            continue
        distances_only = np.array([item[1] for item in sorted_distances_for_token])
        index = np.searchsorted(distances_only, distance)
        close_tokens = [item[0] for item in sorted_distances_for_token[:index]]
        close_distances = np.array([item[1] for item in sorted_distances_for_token[:index]])
        if not close_tokens:
            new_tokens.append(origin_token)  # Keep original token
            continue
        unnormalized_probabilities = np.exp(exp_factor * ((distance - close_distances) / distance))
        total_unnormalized_prob = np.sum(unnormalized_probabilities)
        probabilities = unnormalized_probabilities / total_unnormalized_prob
        selected_token = np.random.choice(close_tokens, p=probabilities)
        new_tokens.append(selected_token)
    sanitized_sent = ' '.join(new_tokens)
    return sanitized_sent

def init_func(epsilon, token_to_vector_dict):
    origin_embed = token_to_vector_dict.get('he', None)
    add_laplace_noise_to_vector(origin_embed, epsilon)