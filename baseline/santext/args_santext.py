import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # Required parameters

    parser.add_argument('--task',
                        choices=['topic_classification','piidocs_classification','synthpai','samsum'],
                        default='storycloze',
                        help='NLP eval tasks')
    
    parser.add_argument('--dataset',
                        choices=['ag_news', "piidocs",'synthpai','samsum'],
                        default='storycloze',
                        help='NLP eval datasets')
    
    parser.add_argument(
        "--data_dir",
        default="./data/",
        type=str,
        help="The input dir"
    )

    parser.add_argument(
        "--output_dir",
        default="./NoiseResults/",
        type=str,
        help="The output dir",
    )

    parser.add_argument(
        "--attack_output_dir",
        default="./NoiseResults_AttackResults/",
        type=str,
        help="The output dir for AttackResults",
    )

    parser.add_argument("--epsilon", type=float, default=1.0, help="privacy parameter epsilon")

    parser.add_argument("--p", type=float, default=0.3, help="SanText+: probability of non-sensitive words to be sanitized")

    parser.add_argument("--sensitive_word_percentage", type=float, default=0.9,
                        help="SanText+: how many words are treated as sensitive")
    

    parser.add_argument("--EmbInver_K", type=int, default=1, help="K for EmbInver Attack")
    parser.add_argument("--MaskInfer_K", type=int, default=1, help="K for MaskInfer Attack")

    parser.add_argument(
        "--bert_model_path",
        default="bert-base-uncased",
        type=str,
        help="bert model name or path"
    )
    
    parser.add_argument(
        '--method',
        choices=['SanText', 'SanText_plus'],
        default='SanText_plus',
        help='Sanitized method'
    )

    parser.add_argument(
        '--embedding_type',
        choices=['glove', 'bert'],
        default='glove',
        help='embedding used for sanitization'
    )

    parser.add_argument(
        "--word_embedding_path",
        default='./data/embeddings/glove_840B-300d.txt',
        type=str,
        help="The pretrained word embedding path. leave it blank if you are using BERT",
    )

    parser.add_argument(
        "--word_embedding_size",
        default=300,
        type=int,
        help="The pretrained word embedding size. leave it blank if you are using BERT",
    )
    
    

    
    return parser
