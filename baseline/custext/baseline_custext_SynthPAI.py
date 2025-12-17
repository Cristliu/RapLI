from func_custext_SynthPAI import *
from args_custext import *
import os
import json
import time
import random
import logging

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
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Running CusText baseline for SynthPAI => task: {args.task}, dataset: {args.dataset}, epsilon: {args.epsilon}")
    
    # Ensure output directory exists
    output_dir = f"./NoiseResults/{args.task}/{args.dataset}" 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load SynthPAI data
    logger.info("Loading SynthPAI data...")
    test_data = load_jsonl_data("./data/synthpai/comments_eval_revised.jsonl")
    logger.info(f"Loaded {len(test_data)} comments from SynthPAI dataset")
    
    # Check if mapping dictionaries already exist
    p_dict_path = f"./data/custext_dict/p_dict_{args.dataset}_eps_{args.epsilon}.txt"
    sim_word_dict_path = f"./data/custext_dict/sim_word_dict_{args.dataset}_eps_{args.epsilon}.txt"

    if os.path.exists(p_dict_path) and os.path.exists(sim_word_dict_path):
        logger.info("Loading existing word mapping dictionaries...")
        with open(p_dict_path, 'r', encoding='utf-8') as dic:
            p_dict = json.load(dic)
        with open(sim_word_dict_path, 'r', encoding='utf-8') as dic:
            sim_word_dict = json.load(dic)
    else:
        logger.info("Creating customized word mapping...")
        start_time = time.time()
        sim_word_dict, p_dict = get_customized_mapping_synthpai(eps=args.epsilon, top_k=args.top_k)
        end_time = time.time()
        get_customized_mapping_time = end_time - start_time
        logger.info(f"get_customized_mapping execution time: {get_customized_mapping_time} seconds")

    # Generate and save perturbed data
    logger.info("Generating perturbed text...")
    start_time = time.time()
    new_data_df = generate_new_sents_synthpai(
        df=test_data, 
        sim_word_dict=sim_word_dict, 
        p_dict=p_dict, 
        save_stop_words=args.save_stop_words
    )
    end_time = time.time()
    generate_new_sents_time = end_time - start_time
    logger.info(f"generate_new_sents execution time: {generate_new_sents_time} seconds")

    # Save customized mapping time (if new mapping was just calculated)
    if 'get_customized_mapping_time' in locals():
        # Ensure Time folder exists
        if not os.path.exists(f"{output_dir}/Time"):
            os.makedirs(f"{output_dir}/Time", exist_ok=True)
            
        map_time_data = {
            "get_customized_mapping_time": get_customized_mapping_time,
        }
        with open(f"{output_dir}/Time/custext_{args.dataset}_epsilon_{args.epsilon}_map_time.json", 'w', encoding='utf-8') as map_time_file:
            json.dump(map_time_data, map_time_file, ensure_ascii=False, indent=4)
    
    # Save perturbation generation time
    time_data = {
        "avg_generate_new_sents_time": generate_new_sents_time/len(test_data)
    }
    with open(f"{output_dir}/custext_epsilon_{args.epsilon}_avg_addnoise_time.json", 'w', encoding='utf-8') as time_file:
        json.dump(time_data, time_file, ensure_ascii=False, indent=4)

    logger.info("CusText processing for SynthPAI completed successfully!")