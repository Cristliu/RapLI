import string
import tiktoken
import random
from sklearn.metrics.pairwise import euclidean_distances
from nltk.corpus import stopwords
from decimal import getcontext
import numpy as np
from transformers import GPT2Tokenizer
import os 
from transformers import logging as transformers_logging
import torch
import json
import tqdm

getcontext().prec = 100  # Set the precision for decimal calculations to 100 digits to ensure high precision, especially when dealing with floating-point numbers.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load GPT2 Tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

stop_words = set(stopwords.words('english'))
# Get all punctuation marks and represent them as set([',', '.']) in a similar form
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
        print(f"\n Warning: Subword lengths are inconsistent: {len(original_subtokens)} vs {len(perturbed_subtokens)}")
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

getcontext().prec = 100  # Set the precision for decimal calculations to 100 digits to ensure high precision, especially when dealing with floating-point numbers.


def get_tokens(text):
    tokenizer = gpt2_tokenizer
    tokens = tokenizer.tokenize(text)
    tokenized_string = tokenizer.convert_tokens_to_string(tokens)
    return tokenized_string

# Calculate the Euclidean distance between two vectors, using np.longdouble type to improve precision. pb is the progress bar object used to display progress.
def calculate_distance(i, j, vector_matrix, pb):
    distance = euclidean_distances(vector_matrix[i].reshape(1, -1).astype(np.longdouble), 
                                   vector_matrix[j].reshape(1, -1).astype(np.longdouble))
    pb.update(1)
    return i, j, distance[0, 0]

punctuation_string = string.punctuation
punctuation_list = list(punctuation_string)

# Generate all possible vector pairs for calculating the distance matrix.
def generate_tasks(n_vectors):
    for i in range(n_vectors):
        for j in range(i + 1, n_vectors):
            yield (i, j)


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
        # Use cdist once to calculate all distances, instead of traversing pair by pair
        distance_matrix = calculate_distance(vector_matrix)
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
    val = epsilon * 19.064721649556482 - 38.1294334077209
    if val > 0:
        # GPU available:
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


# raw_tokens,args.eps,args.model,token_to_vector_dict,sorted_cl100_emb,sen_emb
def perturb_sentence(args, sent, token_to_vector_dict,sorted_distance_data,delta_f_new, bert_tokenizer=None, bert_model=None, device=None):

    # Ensure delta_f_new is a PyTorch tensor
    if not isinstance(delta_f_new, torch.Tensor):
        delta_f_new = torch.tensor(delta_f_new, dtype=torch.float64, device=device)

    enc = tiktoken.encoding_for_model(args.model)
    tokens_b=enc.encode(sent)
    tokens=[(enc.decode_single_token_bytes(t)).decode('Latin-1') for t in tokens_b]
    
    new_tokens=[]
    Delta_u = 1.0  
    exp_factor = args.epsilon / (2 * Delta_u)
    prepro_origin_token = []

    attack_cnt = 0

    EmbInver_success_cnt = 0
    EmbExpStop_success_cnt = 0
    
    EmbInver_success_words = []
    EmbExpStop_success_words = []
    Mask_success_words = []
    Mask_Expstop_success_words = []

    for origin_token in tokens:
        # If it is a number, randomly replace it with a number between 1 and 1000. ### Numbers are not attacked ## because this scheme is very difficult to find embedding
        if(origin_token.isnumeric()):
            random_number = str(random.randint(1, 1000))
            prepro_origin_token.append(origin_token)
            new_tokens.append(random_number)
            continue
        if(origin_token[0]==' '):  # Remove leading spaces
            origin_token=origin_token[1:]
            # If nothing is left after removing the space, skip
            if not origin_token:
                continue
        
        # Get the word vector origin_embed of the original token
        origin_embed = token_to_vector_dict.get(origin_token, None)
        if origin_embed is None:
            new_tokens.append(origin_token)  # Retain the original token  ########### Actually, the original token is directly retained here. Is it considered a successful attack by default? After all, the closest is itself.
            prepro_origin_token.append(origin_token)
            # print(f"Token {origin_token} is keeping the same")

            # If the original word is retained, it is directly considered as a successful attack, because no matter what embedding inversion is used, the closest is itself
            attack_cnt += 1
            EmbInver_success_cnt += 1
            EmbInver_success_words.append(origin_token)
            if origin_token.lower() not in stop_words:
                EmbExpStop_success_cnt += 1
                EmbExpStop_success_words.append(origin_token)
            
            continue
        
        # Add Laplace noise to the word vector to get noise_embed
        noise_embed = add_laplace_noise_to_vector(origin_embed, args.epsilon,delta_f_new)
        # Calculate the distance between the original vector and the noise vector
        distance = np.linalg.norm(origin_embed - noise_embed)

        # Find tokens similar to the original token from the pre-sorted distance data, within the distance range
        sorted_distances_for_token = sorted_distance_data.get(origin_token, None)
        if sorted_distances_for_token is None:
            new_tokens.append(origin_token)  # Retain the original token
            prepro_origin_token.append(origin_token)
            # print(f"Token {origin_token} is keeping the same")

            # If the original word is retained, it is directly considered as a successful attack, because no matter what embedding inversion is used, the closest is itself
            attack_cnt += 1
            EmbInver_success_cnt += 1
            EmbInver_success_words.append(origin_token)
            if origin_token.lower() not in stop_words:
                EmbExpStop_success_cnt += 1
                EmbExpStop_success_words.append(origin_token)

            continue
        distances_only = np.array([item[1] for item in sorted_distances_for_token])
        index = np.searchsorted(distances_only, distance)
        close_tokens = [item[0] for item in sorted_distances_for_token[:index] ]

        close_distances = np.array([item[1] for item in sorted_distances_for_token[:index]])
        if not close_tokens:
            new_tokens.append(origin_token)  # Retain the original token
            prepro_origin_token.append(origin_token)
            # print(f"Token {origin_token} is keeping the same")

            # If the original word is retained, it is directly considered as a successful attack, because no matter what embedding inversion is used, the closest is itself
            attack_cnt += 1
            EmbInver_success_cnt += 1
            EmbInver_success_words.append(origin_token)
            if origin_token.lower() not in stop_words:
                EmbExpStop_success_cnt += 1
                EmbExpStop_success_words.append(origin_token)

            continue
        # Calculate the probability of selecting each replacement word using the exponential mechanism
        unnormalized_probabilities = np.exp(exp_factor * ((distance-close_distances)/distance))
        # print(f"Unnormalized probabilities: {unnormalized_probabilities}")# This calculation is also correct for the probability of the token being selected
        total_unnormalized_prob = np.sum(unnormalized_probabilities)
        probabilities = unnormalized_probabilities / total_unnormalized_prob
        # print(f"Probabilities: {probabilities}")
        # Randomly select a replacement word based on the calculated probability distribution
        selected_token = np.random.choice(close_tokens, p=probabilities)
        # Print the original token and the selected token
        # print(f"Original token: {origin_token}")
        # print(f"Selected token: {selected_token}")
        new_tokens.append(selected_token)
        prepro_origin_token.append(origin_token)


        # Embedding Inversion Attack, check if origin_token is in the top K nearest neighbors of new_tokens, if so, it means the attack is successful
        # Since embedding is a paid query, we see if random_number is in sorted_cl00_embedding.json
        sorted_dis_for_token = sorted_distance_data.get(selected_token, None)
        if sorted_dis_for_token is not None:  # If not found, skip, do not calculate attack or total count
            attack_cnt += 1
            # Directly take the closest K words in sorted_dis_for_token
            close_tokens_K = [item[0] for item in sorted_dis_for_token[:args.EmbInver_K]]
            # print(f"\nOrigin token: {origin_token}")
            # print(f"Close tokens for {selected_token}: {close_tokens_K}")
            # If origin_token is in close_tokens_K, it means the attack is successful # Compare lowercase uniformly
            if origin_token.lower() in [x.lower() for x in close_tokens_K]:
                # print(f"Embedding Inversion Attack successful for (be replaced) token {origin_token}")
                EmbInver_success_cnt += 1
                EmbInver_success_words.append(origin_token)
                if origin_token.lower() not in stop_words:
                    EmbExpStop_success_cnt += 1
                    EmbExpStop_success_words.append(origin_token)
            # else:
            #     print(f"Embedding Inversion Attack failed for (be replaced) token {origin_token}")

    # Calculate attack success rate
    if attack_cnt > 0:
        EmbInver_success_rate = EmbInver_success_cnt / attack_cnt
        EmbExpStop_success_rate = EmbExpStop_success_cnt / attack_cnt

        # print(f"Embedding Inversion Attack success rate: {embInver_rate}")
    
    # Mask Token Inference Attack
    # if bert_tokenizer and bert_model:
    # print("\n===========Mask Token Inference Attack============\n")
    mask_rate, Mask_success_words, mask_rate_ExpStop, Mask_Expstop_success_words = mask_token_inference_attack_subword_topk_batch(
        Mask_success_words, Mask_Expstop_success_words, prepro_origin_token, new_tokens, bert_tokenizer, bert_model,
        device, top_k=args.MaskInfer_K, batch_size=64, stop_words=stop_words
    )
    # else:
    #     mask_rate = 0.0

    sanitized_sent = ' '.join(new_tokens)
    # Print the original sentence and the replaced sentence
    # print(f"Original sentence: {sent}")
    # print(f"Perturbed sentence: {sanitized_sent}")
    return sanitized_sent, EmbInver_success_rate, EmbExpStop_success_rate, EmbInver_success_words, EmbExpStop_success_words, mask_rate, Mask_success_words, mask_rate_ExpStop, Mask_Expstop_success_words



# This function is used to precompute data. If the distance matrix and sensitivity files do not exist, it will call the add_laplace_noise_to_vector function to trigger the precomputation process
def init_func(epsilon,token_to_vector_dict):  
    origin_embed = token_to_vector_dict.get('he', None)
    add_laplace_noise_to_vector(origin_embed,epsilon)
