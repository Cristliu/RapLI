import random
import torch
import numpy as np

def cal_probability_asCustext(word_embed_1, word_embed_2, epsilon=2.0):####### This method does not show significant probability differences when epsilon increases
    distances = torch.norm(word_embed_1.unsqueeze(0) - word_embed_2.unsqueeze(0), dim=2)
    min_dist = distances.min()
    max_dist = distances.max()

    range_dist = max_dist - min_dist
    if range_dist == 0:
        range_dist = 1e-6  # Avoid division by zero
    # Normalize distances    
    new_sim_dist_list = - (distances - min_dist) / range_dist
    # Calculate probability distribution
    tmp = torch.exp(epsilon * new_sim_dist_list / 2) 
    p = tmp / tmp.sum()
    prob_matrix = p.cpu().numpy().tolist()

    prob_matrix = prob_matrix/np.sum(prob_matrix)
    # print(f"prob_matrix after normalization: {prob_matrix}")
    return prob_matrix

def SanText_plus(doc, embedding_matrix, word2id, id2word, sword2id, all_words, p, epsilon, device):
    new_doc = []
    for word in doc:
        if word in word2id:
            # In-vocab
            if word in sword2id:
                # Sensitive Words
                index = word2id[word]
                token_vector = embedding_matrix[index].to(device)
                prob_matrix = cal_probability_asCustext(token_vector, embedding_matrix, epsilon)
                # Convert prob_matrix to numpy format ###cal_probability does not need this
                prob_matrix = np.array(prob_matrix[0])
                sampling_index = np.random.choice(len(prob_matrix), 1, p=prob_matrix)
                ### Print the final selected word
                # print(f"word after sampling: {id2word[sampling_index[0]]}")
                new_doc.append(id2word[sampling_index[0]])
            else:
                # Non-sensitive words
                flip_p = random.random()
                if flip_p <= p:  # 30% chance to be replaced
                    # sample a word from Vs based on prob matrix
                    index = word2id[word]
                    token_vector = embedding_matrix[index].to(device)
                    prob_matrix = cal_probability_asCustext(token_vector, embedding_matrix, epsilon)
                    prob_matrix = np.array(prob_matrix[0])
                    sampling_index = np.random.choice(len(prob_matrix), 1, p=prob_matrix)
                    new_doc.append(id2word[sampling_index[0]])
                else:
                    # keep as the original
                    new_doc.append(word)
        else:
            # Out-of-Vocab words
            sampling_prob = 1 / len(all_words) * np.ones(len(all_words), )
            sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
            new_doc.append(all_words[sampling_index[0]])

    new_doc = " ".join(new_doc)
    return new_doc
