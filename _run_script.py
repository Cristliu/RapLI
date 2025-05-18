import subprocess

TASKS = ['topic_classification','piidocs_classification','samsum'] #'synthpai'is run separately using main_SynthPAI, main_Glove_SynthPAI, baseline_custext_SynthPAI, see _run_script_SynthPAI.py for examples
DATASETS = ['ag_news', "piidocs",'samsum']  # One-to-one correspondence

EPS_VALUES = [0.1, 1.0, 8.0]
K_VALUES = [1, 5]  # EmbInver_K and MaskInfer_K one-to-one correspondence

assert len(TASKS) == len(DATASETS), "TASKS and DATASETS must have the same length and correspond one-to-one."

# Notes: 1. If an error occurs, end the current subprocess and start the next one; 2. Display the progress of batch runs

# Define script paths
BASELINE_SCRIPTS = {
    "santext": {
        "base": "baseline/santext/baseline_SanText.py",
        "attack": "baseline/santext/baseline_SanText_Attacks.py"
    },
    "custext": {
        "base": "baseline/custext/baseline_custext.py",
        "attack": "baseline/custext/baseline_custext_Attacks.py"
    },
    "rantext": {
        "base": "baseline/rantext/baseline_rantext.py",
        "attack": "baseline/rantext/baseline_rantext_Attacks.py"
    },
    "SnD": {
        "base": "baseline/SnD/baseline_SnD.py",
        "attack": "baseline/SnD/baseline_SnD_Attacks.py"
    }
}

OURS_SCRIPTS = {
    "base": "main.py",
    "attack": "main_Attacks.py",
    "glove": "main_Glove.py",
    "glove_attack": "main_Glove_Attacks.py"
}

OURS_ABLATION_SCRIPTS = {
    "base": "main.py",
    "attack": "main_Attacks.py",
    "glove": "main_Glove.py",
    "glove_attack": "main_Glove_Attacks.py"
}

ABLATION_FLAGS = [
    "--ablation_1",
    "--ablation_2",
    "--ablation_3_1",
    "--ablation_3_2",
    "--ablation_3_3",
    "--ablation_4",
]

PRIVACY_ANALYSIS_SCRIPTS = [
    "Privacy_Analysis/mask_inference_attack_for_unsanitized.py",
    "Privacy_Analysis/Privacy_Similarity_3gram_PII.py",
    "Privacy_Analysis/Further_Attack_PII_match_Opti_multi.py"
]

def run_script(cmd, description):
    """
    Run a subprocess command and handle possible exceptions.
    """
    try:
        print(f"Starting: {description}")
        subprocess.run(cmd, check=True)
        print(f"Successfully ran: {description}\n")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run: {description}\nError message: {e}\n")
    except Exception as e:
        print(f"Unknown error while running: {description}\nError message: {e}\n")

def run_baseline():
    """
    Run the Baseline scripts.
    """
    print("\n=== Running Baseline Scripts ===\n")

    for task, dataset in zip(TASKS, DATASETS):
        print(f"Processing task: {task}, dataset: {dataset}")
        for script_type, scripts in BASELINE_SCRIPTS.items():
            # Run base scripts, iterate over EPS_VALUES
            base_script = scripts["base"]
            for eps in EPS_VALUES:
                description = f"Baseline {script_type} base script: {base_script} (task: {task}, dataset: {dataset}, ε: {eps})"
                cmd = ["python", base_script, "--task", task, "--dataset", dataset, "--epsilon", str(eps)]
                run_script(cmd, description)
            
            # Run attack scripts
            attack_script = scripts["attack"]
            for eps in EPS_VALUES:
                for k in K_VALUES:
                    description = f"Baseline {script_type} attack script: {attack_script} (task: {task}, dataset: {dataset}, ε: {eps}, K: {k})"
                    cmd = [
                        "python", attack_script,
                        "--task", task,
                        "--dataset", dataset,
                        "--epsilon", str(eps),
                        "--EmbInver_K", str(k),
                        "--MaskInfer_K", str(k)
                    ]
                    run_script(cmd, description)

def run_ours():
    """
    Run the Ours scripts.
    """
    print("\n=== Running Ours Scripts ===\n")
    for task, dataset in zip(TASKS, DATASETS):
        print(f"Processing task: {task}, dataset: {dataset}")
        # Run base scripts
        base_script = OURS_SCRIPTS["base"]
        for eps in EPS_VALUES:
            description = f"Ours base script: {base_script} (task: {task}, dataset: {dataset}, ε: {eps})"
            cmd = [
                "python", base_script,
                "--task", task,
                "--dataset", dataset,
                "--epsilon", str(eps)
            ]
            run_script(cmd, description)
        
        # Run attack scripts
        attack_script = OURS_SCRIPTS["attack"]
        for eps in EPS_VALUES:
            for k in K_VALUES:
                description = f"Ours attack script: {attack_script} (task: {task}, dataset: {dataset}, ε: {eps}, K: {k})"
                cmd = [
                    "python", attack_script,
                    "--task", task,
                    "--dataset", dataset,
                    "--epsilon", str(eps),
                    "--EmbInver_K", str(k),
                    "--MaskInfer_K", str(k)
                ]
                run_script(cmd, description)
        
        # Run Glove base scripts
        glove_script = OURS_SCRIPTS["glove"]
        for eps in EPS_VALUES:
            description = f"Ours Glove base script: {glove_script} (task: {task}, dataset: {dataset}, ε: {eps})"
            cmd = [
                "python", glove_script,
                "--task", task,
                "--dataset", dataset,
                "--epsilon", str(eps)
            ]
            run_script(cmd, description)
        
        # Run Glove attack scripts
        glove_attack_script = OURS_SCRIPTS["glove_attack"]
        for eps in EPS_VALUES:
            for k in K_VALUES:
                description = f"Ours Glove attack script: {glove_attack_script} (task: {task}, dataset: {dataset}, ε: {eps}, K: {k})"
                cmd = [
                    "python", glove_attack_script,
                    "--task", task,
                    "--dataset", dataset,
                    "--epsilon", str(eps),
                    "--EmbInver_K", str(k),
                    "--MaskInfer_K", str(k)
                ]
                run_script(cmd, description)

def run_ours_ablation():
    """
    Run the Ours Ablation Experiments scripts.
    Only for task=topic_classification, dataset=ag_news
    """
    print("\n=== Running Ours Ablation Experiments Scripts ===\n")
    # task = 'topic_classification'
    # dataset = 'ag_news'
    # EPS_VALUES = [1.0]
    
    for ablation_flag in ABLATION_FLAGS:
        for eps in EPS_VALUES:
            # Run base Ablation scripts
            base_script = "main.py"
            description = f"Ours Ablation base script: {base_script} (task: {task}, dataset: {dataset}, ε: {eps}, flag: {ablation_flag})"
            cmd = [
                "python", base_script,
                "--task", task,
                "--dataset", dataset,
                "--epsilon", str(eps),
                ablation_flag
            ]
            run_script(cmd, description)
            
            # Run attack Ablation scripts
            attack_script = "main_Attacks.py"
            description = f"Ours Ablation attack script: {attack_script} (task: {task}, dataset: {dataset}, ε: {eps}, K: {K_VALUES}, flag: {ablation_flag})"
            for k in K_VALUES:
                cmd = [
                    "python", attack_script,
                    "--task", task,
                    "--dataset", dataset,
                    "--epsilon", str(eps),
                    "--EmbInver_K", str(k),
                    "--MaskInfer_K", str(k),
                    ablation_flag
                ]
                run_script(cmd, description)

def run_privacy_analysis():
    """
    Run the Privacy Analysis scripts.
    """
    print("\n=== Running Privacy Analysis Scripts ===\n")
    for script in PRIVACY_ANALYSIS_SCRIPTS:
        description = f"Privacy Analysis script: {script}"
        cmd = ["python", script]
        run_script(cmd, description)

def main():
    
    # Run Baseline scripts
    run_baseline()
    
    # Run Ours scripts
    run_ours()
    
    # Run Ours Ablation Experiments scripts
    run_ours_ablation()
    
    # Run Privacy Analysis scripts
    run_privacy_analysis()
    

if __name__ == "__main__":
    main()
