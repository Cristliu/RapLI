import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # Required parameters

    parser.add_argument('--task',
                        choices=['topic_classification','piidocs_classification','synthpai','samsum'],
                        default='samsum',
                        help='NLP eval tasks')
    
    parser.add_argument('--dataset',
                        choices=['ag_news', "piidocs",'synthpai','samsum'],
                        default='samsum',
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

    parser.add_argument("--EmbInver_K", type=int, default=1, help="K for EmbInver Attack")
    parser.add_argument("--MaskInfer_K", type=int, default=1, help="K for MaskInfer Attack")

    return parser
