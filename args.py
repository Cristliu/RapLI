import argparse

def get_parser():

    parser = argparse.ArgumentParser()

    # Required parameters

    parser.add_argument('--task',
                        choices=['topic_classification', 'piidocs_classification','synthpai','samsum'],
                        default='samsum',
                        help='NLP eval tasks')
    
    parser.add_argument('--dataset',
                        choices=['ag_news', "piidocs",'synthpai','samsum'],
                        default='samsum',
                        help='NLP eval datasets')
    
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert/distilbert-base-cased",
        help="HuggingFace model name."
    )
    
    parser.add_argument(
        "--epsilon",
        default=0.1,
        type=float,
        help="The minimum privacy budget."
    )
    parser.add_argument(
        "--max_budget",
        default=8.0,
        type=float,
        help="The maximum privacy budget."
    )

    parser.add_argument(
        "--EmbInver_K",
        type=int,
        default=1,
        help="K for EmbInver Attack"
    )
    parser.add_argument(
        "--MaskInfer_K",
        type=int,
        default=1,
        help="K for MaskInfer Attack"
    )


    # For ablation experiments, add switch parameters respectively,
    # The default is False, indicating that the corresponding experiment is not enabled
    parser.add_argument("--ablation_1", action="store_true", help="Ablation experiment 1 - args.model_name directly uses bert-base-uncased.")
    parser.add_argument("--ablation_2", action="store_true", help="Ablation experiment 2 - do not calculate epsilon_S")
    parser.add_argument("--ablation_3_1", action="store_true", help="Ablation experiment 3_1 - K_base is set to 10.")
    parser.add_argument("--ablation_3_2", action="store_true", help="Ablation experiment 3_1 - K_base is set to 50.")
    parser.add_argument("--ablation_3_3", action="store_true", help="Ablation experiment 3_1 - K_base is set to 100.")
    parser.add_argument("--ablation_4", action="store_true", help="Ablation experiment 4 - do not perturb tokens identified as high risk.")

    return parser