import subprocess

# Define tasks and datasets (assumed to be one-to-one matching)
TASKS = ['synthpai']
DATASETS = ['synthpai']  # one-to-one correspondence

# Define ε values and K values
EPS_VALUES = [0.1, 1.0, 8.0]
K_VALUES = [1, 5]  # EmbInver_K and MaskInfer_K one-to-one correspondence

# Ensure TASKS and DATASETS have the same length
assert len(TASKS) == len(DATASETS), "TASKS and DATASETS must have the same length and correspond one-to-one."


BASELINE_SCRIPTS = {
    "custext": {
        "base": "baseline/custext/baseline_custext_SynthPAI.py",
        "attack": "baseline/custext/baseline_custext_Attacks_SynthPAI.py"
    }
}

OURS_SCRIPTS = {
    "base": "main_SynthPAI.py",
    "attack": "main_SynthPAI_Attacks.py"
}


# Define Privacy Analysis scripts
PRIVACY_ANALYSIS_SCRIPTS = [
    "Privacy_Analysis/Privacy_Similarity_3gram_PII.py",
    "Privacy_Analysis/Further_Attack_PII_match_Opti_multi.py"
]



def run_script(cmd, description):
    """
    Run a subprocess command and handle possible exceptions.
    """
    try:
        print(f"Starting to run: {description}")
        subprocess.run(cmd, check=True)
        print(f"Successfully run: {description}\n")
    except subprocess.CalledProcessError as e:
        print(f"Run failed: {description}\nError message: {e}\n")
    except Exception as e:
        print(f"Unknown error while running: {description}\nError message: {e}\n")


def run_baseline():
    """
    Run Baseline scripts.
    """
    print("\n=== Running Baseline Scripts ===\n")

    TASKS = ['synthpai']
    DATASETS = ['synthpai']  # one-to-one correspondence

    for task, dataset in zip(TASKS, DATASETS):
        print(f"Processing task: {task}, dataset: {dataset}")
        for script_type, scripts in BASELINE_SCRIPTS.items():
            # Run base script, iterate through EPS_VALUES
            base_script = scripts["base"]
            for eps in EPS_VALUES:
                description = f"Baseline {script_type} base script: {base_script} (task: {task}, dataset: {dataset}, ε: {eps})"
                cmd = ["python", base_script, "--task", task, "--dataset", dataset, "--epsilon", str(eps)]
                run_script(cmd, description)
            
            # Run attack script
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
    Run Our scripts.
    """
    print("\n=== Running Our Scripts ===\n")
    for task, dataset in zip(TASKS, DATASETS):
        print(f"Processing task: {task}, dataset: {dataset}")
        # Run base script
        base_script = OURS_SCRIPTS["base"]
        for eps in EPS_VALUES:
            description = f"Our base script: {base_script} (task: {task}, dataset: {dataset}, ε: {eps})"
            cmd = [
                "python", base_script,
                "--task", task,
                "--dataset", dataset,
                "--epsilon", str(eps)
            ]
            run_script(cmd, description) 


    # Run attack script
        attack_script = OURS_SCRIPTS["attack"]
        for eps in EPS_VALUES:
            for k in K_VALUES:
                description = f"Our attack script: {attack_script} (task: {task}, dataset: {dataset}, ε: {eps}, K: {k})"
                cmd = [
                    "python", attack_script,
                    "--task", task,
                    "--dataset", dataset,
                    "--epsilon", str(eps),
                    "--EmbInver_K", str(k),
                    "--MaskInfer_K", str(k)
                ]
                run_script(cmd, description)


def run_privacy_analysis():
    """
    Run Privacy Analysis scripts.
    """
    print("\n=== Running Privacy Analysis Scripts ===\n")
    for script in PRIVACY_ANALYSIS_SCRIPTS:
        description = f"Privacy Analysis script: {script}"
        cmd = ["python", script]
        run_script(cmd, description)


def main():

    # Run Baseline part
    run_baseline()
    
    # Run Our part
    run_ours()

    # # Run Privacy Analysis part
    run_privacy_analysis()
    

if __name__ == "__main__":
    main()