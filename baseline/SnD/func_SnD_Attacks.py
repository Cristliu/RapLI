import torch
import numpy as np
from func_SnD import sample_noise_L2_lap, get_token_embedding, get_closest_token
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

def SnD_plus(
    args, doc_tokens, original_text, label, hypothesis,
    tokenizer, bert_model, bert_mask_model, stop_words, device, dataset="ag_news"
):
    # ==============  SnD perturbation  ==============
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(doc_tokens), device=device)
    init_emb = get_token_embedding(input_ids, bert_model, device)

    # Add noise (adjustable user_scale / or not)
    noises = sample_noise_L2_lap(init_emb.shape, eta=args.epsilon, device=device, user_scale=0.01)
    noise_init_emb = init_emb + noises

    perturbed_subtokens = []
    for i in range(len(doc_tokens)):
        new_tk = get_closest_token(noise_init_emb[i], tokenizer, bert_model, device)
        perturbed_subtokens.append(new_tk)

    # ==============  Embedding Inversion Attack  ==============
    EmbIver_success_words = []
    EmbExpStop_success_words = []
    total = len(doc_tokens)
    success = 0
    EmbExpStop_success = 0
    EmbInver_K = args.EmbInver_K
    vocab_ids = torch.tensor(list(tokenizer.get_vocab().values()), device=device)
    vocab_embs = get_token_embedding(vocab_ids, bert_model, device)  # [vocab_size, d_emb]

    for ori,new in zip(doc_tokens, perturbed_subtokens):
        w_id = tokenizer.convert_tokens_to_ids([new])
        w_emb = get_token_embedding(torch.tensor(w_id, device=device), bert_model, device).squeeze(0)  # [d_emb]

        dist = torch.norm(vocab_embs - w_emb.unsqueeze(0), p=2, dim=1)
        topk_indices = dist.topk(EmbInver_K, largest=False).indices
        topk_tokens = [vocab_ids[idx].item() for idx in topk_indices]
        topk_tokens_str = [tokenizer.convert_ids_to_tokens(x).lower() for x in topk_tokens]

        if ori.lower() in topk_tokens_str:
            success += 1
            EmbIver_success_words.append(ori)
            if ori.lower() not in stop_words:
                EmbExpStop_success += 1
                EmbExpStop_success_words.append(ori)

    embInver_rate = success / total if total > 0 else 0.0
    embExpStop_rate = EmbExpStop_success / total if total > 0 else 0.0

    # ==============  Mask Inference Attack  ==============
    mask_rate, Mask_success_words, mask_rate_ExpStop, Mask_Expstop_success_words = mask_token_inference_attack_subword_topk_batch(
        [], [], doc_tokens, perturbed_subtokens, tokenizer, bert_mask_model, device,
        top_k=args.MaskInfer_K, batch_size=32, stop_words=stop_words
    )

    if dataset != "mednli":
        hypothesis = None

    return {
        "original_text": original_text,
        "label": label,
        "hypothesis": hypothesis,
        "perturbed_sentence": tokenizer.convert_tokens_to_string(perturbed_subtokens),
        "embInver_rate": embInver_rate,
        "embExpStop_rate": embExpStop_rate,
        "embInver_success_words": EmbIver_success_words,
        "embExpStop_success_words": EmbExpStop_success_words,
        "mask_rate": mask_rate,
        "mask_rate_ExpStop": mask_rate_ExpStop,
        "mask_success_words": Mask_success_words,
        "mask_Expstop_success_words": Mask_Expstop_success_words,
    }