import argparse

def get_parser():

    parser = argparse.ArgumentParser()

    # Required parameters

    parser.add_argument('--task',
                        choices=['topic_classification', 'piidocs_classification','synthpai','samsum','spam_email_classification'],
                        default='samsum',
                        help='NLP eval tasks')
    
    parser.add_argument('--dataset',
                        choices=['ag_news', "piidocs",'synthpai','samsum','spam_email'],
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

    parser.add_argument("--new_ablation_1", action="store_true", help="Remove PII detection and risk assessment; use epsilon for all tokens,epsilon=1")
    parser.add_argument("--new_ablation_2", action="store_true", help="All tokens undergo score reversal. (sim_scores[::-1])")

    # New: Chinese PII detection related parameters
    parser.add_argument("--use_transformer_pii", action="store_true", 
                       help="Use Transformer-enhanced PII detector for Chinese datasets")
    parser.add_argument("--pii_detection_mode", type=str, 
                       choices=["traditional", "intelligent", "transformer"], 
                       default="intelligent",
                       help="PII detection mode: traditional (Flair+Presidio), intelligent (rule-based), transformer (Transformer-enhanced)")
    
    # New: Fine-grained non-PII token perturbation control parameter
    parser.add_argument("--non_pii_perturbation_prob", type=float, 
                       choices=[0.1, 0.3, 0.5, 0.7], 
                       default=0.3,
                       help="Perturbation probability for non-PII nouns and verbs (0.1, 0.3, 0.5, 0.7)")

    return parser