from func_custext import *
from args_custext import *
import os
import json
import time
import random

def set_random_seed(seed_value=42):
    """
    Set random seed to ensure reproducible results.
    """
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

    output_dir = f"./NoiseResults/{args.task}/{args.dataset}" #custext_epsilon_{args.epsilon}
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
        print(f"get_customized_mapping execution time: {get_customized_mapping_time} seconds")


    if args.privatization_strategy == "s1":
        start_time = time.time()
        new_data_df = generate_new_sents_s1(df=test_data, sim_word_dict=sim_word_dict, p_dict=p_dict, save_stop_words=args.save_stop_words, type="test")
        end_time = time.time()
        generate_new_sents_s1_time = end_time - start_time
        print(f"generate_new_sents_s1 execution time: {generate_new_sents_s1_time} seconds")


    #If get_customized_mapping_time exists
    if 'get_customized_mapping_time' in locals():
        map_time_data = {
            "get_customized_mapping_time": get_customized_mapping_time,
        }

        #Ensure {output_dir}/Time folder exists
        if not os.path.exists(f"{output_dir}/Time"):
            os.makedirs(f"{output_dir}/Time")

        with open(f"{output_dir}/Time/custext_{args.dataset}_epsilon_{args.epsilon}_map_time.json", 'w', encoding='utf-8') as map_time_file:
            json.dump(map_time_data, map_time_file, ensure_ascii=False, indent=4)
    
    time_data = {
        # "get_customized_mapping_time": get_customized_mapping_time,
        "avg_generate_new_sents_s1_time": generate_new_sents_s1_time/len(test_data)
    }

    with open(f"{output_dir}/custext_epsilon_{args.epsilon}_avg_addnoise_time.json", 'w', encoding='utf-8') as time_file:
        json.dump(time_data, time_file, ensure_ascii=False, indent=4)