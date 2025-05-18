import random
import torch
import numpy as np
from transformers import logging as transformers_logging



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


def cal_probability_asCustext(word_embed_1, word_embed_2, epsilon=2.0):
    """
    Calculate probability distribution based on distance for sampling replacement words.
    """
    distances = torch.norm(word_embed_1.unsqueeze(0) - word_embed_2.unsqueeze(0), dim=2)
    min_dist = distances.min()
    max_dist = distances.max()
    range_dist = max_dist - min_dist
    if range_dist == 0:
        range_dist = 1e-6  # Avoid division by zero

    # Normalize distance and convert to similarity
    sim_dist = - (distances - min_dist) / range_dist

    # Calculate probability distribution
    tmp = torch.exp(epsilon * sim_dist / 2) 
    p = tmp / tmp.sum()
    prob_matrix = p.cpu().numpy().tolist()
    prob_matrix = prob_matrix/np.sum(prob_matrix)


    return prob_matrix


def cal_probability_asCustext_KNN(word_embed_1, word_embed_2, K=10):
    """
    Return the indices of the K vectors closest to the sampled_vector.
    """
    distances = torch.norm(word_embed_1.unsqueeze(0) - word_embed_2.unsqueeze(0), dim=2)
    _, indices = torch.topk(distances, K, largest=False)
    return indices.cpu().numpy()


def SanText_plus(args, doc, embedding_matrix, word2id, id2word, sword2id,
               all_words, p, epsilon, device, bert_tokenizer=None, bert_model=None, EmbInver_K=1,stop_words=None):
    new_doc = []
    EmbIver_success_words = []
    EmbExpStop_success_words = []
    Mask_success_words = []
    Mask_Expstop_success_words = []
    success = 0
    EmbExpStop_success = 0
    total = len(doc)

    
    for word in doc:
        if word in word2id:
            if word in sword2id:
                # Sensitive Words
                index = word2id[word]
                token_vector = embedding_matrix[index].to(device)  # (1, D)
                prob_matrix = cal_probability_asCustext(token_vector, embedding_matrix, epsilon)
                prob_matrix = np.array(prob_matrix[0])
                sampling_index = np.random.choice(len(prob_matrix), 1, p=prob_matrix)[0]
                new_doc.append(id2word[sampling_index])
                
                # Embedding Inversion Attack
                
                # if word.lower() not in stop_words:
                sampled_vector = embedding_matrix[sampling_index].to(device)  # (1, D)
                indices = cal_probability_asCustext_KNN(sampled_vector, embedding_matrix, K=EmbInver_K)
                K_words = [id2word[i] for i in indices[0]]
                if word.lower() in [w.lower() for w in K_words]:
                    success += 1
                    EmbIver_success_words.append(word)
                    if word.lower() not in stop_words:
                        EmbExpStop_success += 1
                        EmbExpStop_success_words.append(word)
            else:
                # Non-sensitive words
                if random.random() <= p:
                    index = word2id[word]
                    token_vector = embedding_matrix[index].to(device)  # (1, D)
                    prob_matrix = cal_probability_asCustext(token_vector, embedding_matrix, epsilon)
                    prob_matrix = np.array(prob_matrix[0])
                    sampling_index = np.random.choice(len(prob_matrix), 1, p=prob_matrix)[0]
                    new_doc.append(id2word[sampling_index])

                    # Embedding Inversion Attack
                    # if word.lower() not in stop_words:
                    sampled_vector = embedding_matrix[sampling_index].to(device)  # (1, D)
                    indices = cal_probability_asCustext_KNN(sampled_vector, embedding_matrix, K=EmbInver_K)
                    K_words = [id2word[i] for i in indices[0]]
                    if word.lower() in [w.lower() for w in K_words]:
                        success += 1
                        EmbIver_success_words.append(word)
                        if word.lower() not in stop_words:
                            EmbExpStop_success += 1
                            EmbExpStop_success_words.append(word)

                else:
                    new_doc.append(word)

                    # if word.lower() not in stop_words:
                    # Embedding Inversion Attack on original word ###Because the attacker does not know whether the replacement was performed in this random process, they must attack all words
                    original_vector = embedding_matrix[word2id[word]].to(device)
                    indices = cal_probability_asCustext_KNN(original_vector, embedding_matrix, K=EmbInver_K)
                    K_words = [id2word[i] for i in indices[0]]
                    if word.lower() in [w.lower() for w in K_words]:
                        success += 1
                        EmbIver_success_words.append(word)
                        if word.lower() not in stop_words:
                            EmbExpStop_success += 1
                            EmbExpStop_success_words.append(word)

        else: #This attack is also correct because there is a probability of replacement
            # Out-of-Vocab words, randomly sample replacements.
            # print(f"Warning: OOV word: {word}")
            # sampling_prob = np.ones(len(all_words)) / len(all_words)
            # Should not use the length of all_words (40819), should use the length of the vocabulary (28086)
            sampling_prob = np.ones(len(word2id)) / len(word2id)
            # print(f"Sampling prob shape: {sampling_prob.shape}")
            sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)[0]
            # print(f"Sampling index: {sampling_index}")
            new_doc.append(all_words[sampling_index])
            
            # Embedding Inversion Attack
            sampled_vector = embedding_matrix[sampling_index].to(device)
            indices = cal_probability_asCustext_KNN(sampled_vector, embedding_matrix, K=EmbInver_K)
            K_words = [id2word[i] for i in indices[0]]
            if word.lower() in [w.lower() for w in K_words]:
                success += 1
                EmbIver_success_words.append(word)
                if word.lower() not in stop_words:
                    EmbExpStop_success += 1
                    EmbExpStop_success_words.append(word)


    # Mask Token Inference Attack
    if bert_tokenizer and bert_model:
        mask_rate, Mask_success_words, mask_rate_ExpStop, Mask_Expstop_success_words = mask_token_inference_attack_subword_topk_batch(
            Mask_success_words, Mask_Expstop_success_words, doc, new_doc, bert_tokenizer, bert_model,
            device, top_k=args.MaskInfer_K, batch_size=64, stop_words=stop_words
        )
    else:
        mask_rate = 0.0

    new_doc_str = " ".join(new_doc)
    embInver_rate = success / total if total > 0 else 0.0
    embExpStop_rate = EmbExpStop_success / total if total > 0 else 0.0

    return new_doc_str, embInver_rate, embExpStop_rate, mask_rate, mask_rate_ExpStop, EmbIver_success_words, EmbExpStop_success_words, Mask_success_words, Mask_Expstop_success_words
