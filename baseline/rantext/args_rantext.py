import argparse

def get_parser():
    parser = argparse.ArgumentParser() 

    parser.add_argument('--task',
                        choices=['topic_classification', 'piidocs_classification','synthpai','samsum'],
                        default='topic_classification',
                        help='NLP eval tasks')
    parser.add_argument('--dataset',
                        choices=['ag_news', "piidocs",'synthpai','samsum'],
                        default='ag_news',
                        help='NLP eval datasets')

    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--EmbInver_K", type=int, default=1)
    parser.add_argument("--MaskInfer_K", type=int, default=1)

    parser.add_argument(
        "--bert_model_path",
        default="bert-base-uncased",
        type=str,
        help="bert model name or path"
    )
    return parser