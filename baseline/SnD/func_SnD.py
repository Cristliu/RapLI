import torch
import math

########################################
# 1.  Sample \ell_2-Laplace noise function
########################################
def sample_noise_L2_lap(d_shape, eta, device, user_scale=1.0):
    """
    Given d_shape=(..., d_emb), sample \ell_2-Laplace noise in the d_emb dimension.
    When eta is larger, the noise is smaller. Sampling formula:
      1) l ∼ Gamma(shape=d_emb, scale=1/eta)  equivalent to  l ∼ Gamma(concentration=d_emb, rate=eta)
      2) v ∼ UniformSphere(d_emb)
      3) z = l * v

    """

    # prefix could be batch_size, seq_len, etc.
    *prefix, d_emb = d_shape
    total_size = math.prod(prefix)  # Python 3.8+ has math.prod

    # ----------------------------
    # Step 1: l ∼ Gamma(d_emb, 1/eta) 
    #         In older versions (or versions that do not support the scale parameter) of PyTorch, use (concentration, rate)
    #         => rate=eta, mean= shape / rate = d_emb / eta
    # ----------------------------
    alpha = torch.tensor(float(d_emb), dtype=torch.float32, device=device)  # concentration
    beta  = torch.tensor(eta, dtype=torch.float32, device=device)           # rate
    gamma_dist = torch.distributions.gamma.Gamma(
        concentration=alpha,
        rate=beta
    )
    # Sample total_size scalars  => shape=[total_size]
    l_vals = gamma_dist.sample((total_size,)).to(device)

    # ----------------------------
    # Step 2: v ∼ UniformSphere(d_emb)
    # ----------------------------
    normal_samples = torch.randn(total_size, d_emb, device=device)
    norms = torch.norm(normal_samples, p=2, dim=1, keepdim=True)
    v = normal_samples / norms  # unit vector

    # ----------------------------
    # Step 3: z = l_vals * v
    # ----------------------------
    z = v * l_vals.unsqueeze(-1)  # [total_size, d_emb]

    # Artificially add a scaling factor ==> for comparison with other baselines
    z = z * user_scale
    z = z.view(*prefix, d_emb)

    return z


########################################
# 2.  Get token embedding
########################################
def get_token_embedding(token_ids, model, device):
    """
    Given several token ids, return their BERT embeddings
    token_ids: Tensor(...), already on the corresponding device
    """
    with torch.no_grad():
        # BERT's token embedding layer
        emb = model.embeddings.word_embeddings(token_ids)
    return emb

########################################
# 3.  Find the token closest to a given vector
########################################
def get_closest_token(embedding, tokenizer, model, device):
    """
    Given a d_emb dimensional embedding, 
    find the token in the entire vocab that is closest (smallest Euclidean distance).
    """
    vocabulary = tokenizer.get_vocab()
    # vocab_size = len(vocabulary)
    token_ids = torch.tensor(list(vocabulary.values()), device=device)  # [vocab_size]
    
    # Get embeddings for all vocab tokens
    word_embeddings = get_token_embedding(token_ids, model, device)     # [vocab_size, d_emb]

    # Expand embedding to the same shape as word_embeddings for comparison
    # embedding is [d_emb], reshape to (1,d_emb), then broadcast to (vocab_size, d_emb)
    embedding = embedding.unsqueeze(0).expand(word_embeddings.size())   # [vocab_size, d_emb]

    # Calculate Euclidean distance
    distance = torch.norm(embedding - word_embeddings, p=2, dim=1)      # [vocab_size]
    closest_idx = distance.argmin()
    closest_token_id = token_ids[closest_idx].item()

    # Return the actual token (str)
    return tokenizer.convert_ids_to_tokens(closest_token_id)

########################################
# 4.  Main function: SnD
########################################
def SnD(doc, tokenizer, model, epsilon, device):
    """
    Given a doc (list of tokens), 
    add \ell_2-Laplace noise and project back to the nearest token.

    doc:         list[str], original tokenized result
    tokenizer:   BertTokenizer
    model:       BertModel
    epsilon:     Here we treat it as η, just ensure that the larger it is, the smaller the noise
    device:      'cpu' or 'cuda'

    return:  the sentence after perturbation (string)
    """
    # First convert doc to ids
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(doc), device=device)
    # Get original embeddings
    init_emb = get_token_embedding(input_ids, model, device)   # shape=[seq_len, d_emb]

    # Sample noise
    # Note: epsilon here is equivalent to "η" in the paper
    noises = sample_noise_L2_lap(init_emb.shape, eta=epsilon, device=device,user_scale=0.01)
    #######0.05 when epsilon=8 will not be completely equal to the original word; but epsilon-1 is not very effective
    #######Using 0.01, epsilon=1 is at least somewhat close, although epsilon=8 is almost identical

    # Add noise
    noise_init_emb = init_emb + noises

    # For each token embedding, find the nearest word
    perturbed_tokens = []
    for i in range(len(doc)):
        perturbed_token = get_closest_token(noise_init_emb[i], tokenizer, model, device)
        perturbed_tokens.append(perturbed_token)

    # Concatenate into a string using Tokenizer's convert_tokens_to_string method
    new_doc = tokenizer.convert_tokens_to_string(perturbed_tokens)
    return new_doc

