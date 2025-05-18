from func_custext_Attacks import *
from args_custext import *
import os
import json
import time
import random
from transformers import BertTokenizer, BertForMaskedLM


def set_random_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    set_random_seed(42)
    parser = get_parser()
    args = parser.parse_args()
    
    test_data = load_data(args.dataset)

    output_dir = f"./NoiseResults_AttackResults/{args.task}/{args.dataset}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    p_dict_path = f"./data/custext_dict/p_dict_{args.dataset}_eps_{args.epsilon}.txt"
    sim_word_dict_path = f"./data/custext_dict/sim_word_dict_{args.dataset}_eps_{args.epsilon}.txt"

    if os.path.exists(p_dict_path) and os.path.exists(sim_word_dict_path):
        with open(p_dict_path, 'r', encoding='utf-8') as dic:
            p_dict = json.load(dic)
        with open(sim_word_dict_path, 'r', encoding='utf-8') as dic:
            sim_word_dict = json.load(dic)
    else:
        start_time = time.time()
        sim_word_dict, p_dict = get_customized_mapping(eps=args.epsilon, top_k=args.top_k)
        end_time = time.time()
        get_customized_mapping_time = end_time - start_time
        print(f"get_customized_mapping runtime: {get_customized_mapping_time} seconds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load BERT model and tokenizer once
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    bert_model = BertForMaskedLM.from_pretrained(args.bert_model_path)
    bert_model.to(device)
    bert_model.eval()

    if args.privatization_strategy == "s1":
        start_time = time.time()
        new_data_df = generate_new_sents_s1(df=test_data, sim_word_dict=sim_word_dict, p_dict=p_dict, save_stop_words=args.save_stop_words, type="test", EmbInver_K=args.EmbInver_K, MaskInfer_K = args.MaskInfer_K, bert_tokenizer=bert_tokenizer, bert_model=bert_model)
        end_time = time.time()
        generate_new_sents_s1_time = end_time - start_time
        print(f"generate_new_sents_with_attack runtime: {generate_new_sents_s1_time} seconds")


    # If get_customized_mapping_time exists
    if 'get_customized_mapping_time' in locals():
        print(f"warning: get_customized_mapping_time exists")

        map_time_data = {
            "get_customized_mapping_time": get_customized_mapping_time,
        }

        # Ensure {output_dir}/Time directory exists
        if not os.path.exists(f"{output_dir}/Time"):
            os.makedirs(f"{output_dir}/Time")

        with open(f"{output_dir}/Time/custext_{args.dataset}_epsilon_{args.epsilon}_map_time.json", 'w', encoding='utf-8') as map_time_file:
            json.dump(map_time_data, map_time_file, ensure_ascii=False, indent=4)
    
    time_data = {
        "avg_generate_new_sents_s1_time": generate_new_sents_s1_time/len(test_data)
    }

    with open(f"{output_dir}/custext_epsilon_{args.epsilon}_EmbInver_{args.EmbInver_K}_MaskInfer_{args.MaskInfer_K}_avg_addnoise_with_attack_time.json", 'w', encoding='utf-8') as time_file:
        json.dump(time_data, time_file, ensure_ascii=False, indent=4)

    # Calculate the average success rate of Embedding Inversion Attack, i.e., the average value of 'success_cnt' in new_data_df, and save it to the corresponding json file
    avg_embInver_rate = new_data_df['embInver_rate'].mean()
    avg_embExpStop_rate = new_data_df['embExpStop_rate'].mean()
    avg_mask_rate = new_data_df['mask_rate'].mean()
    avg_mask_rate_ExpStop = new_data_df['mask_rate_ExpStop'].mean()

    avg_success_rate_data = {
        "avg_embInver_rate": avg_embInver_rate,
        "avg_embExpStop_rate": avg_embExpStop_rate,
        "avg_mask_rate": avg_mask_rate,
        "avg_mask_rate_ExpStop": avg_mask_rate_ExpStop
    }
    with open(f"{output_dir}/custext_epsilon_{args.epsilon}_EmbInver_{args.EmbInver_K}_MaskInfer_{args.MaskInfer_K}_avg_success_rate_data.json", 'w', encoding='utf-8') as success_rate_file:
        json.dump(avg_success_rate_data, success_rate_file, ensure_ascii=False, indent=4)
