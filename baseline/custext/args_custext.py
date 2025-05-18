import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--task',
                        choices=['topic_classification','piidocs_classification','synthpai','samsum'],
                        default='topic_classification',
                        help='NLP eval tasks')
    parser.add_argument('--dataset',
                        choices=['ag_news', "piidocs",'synthpai','samsum'],
                        default='ag_news',
                        help='NLP eval datasets')
    
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--EmbInver_K", type=int, default=1)
    parser.add_argument("--MaskInfer_K", type=int, default=1)

    parser.add_argument(
        "--bert_model_path",
        default="bert-base-uncased",
        type=str,
        help="bert model name or path"
    )

    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--mapping_strategy", type=str, default="conservative")
    parser.add_argument("--privatization_strategy", type=str, default="s1")
    parser.add_argument("--save_stop_words", type=bool, default=True)
    return parser
