# Code for Rap-LI


## 🛠️ Installation

### 1. Set up the `Rap_part1` environment

```bash
conda create -n Rap_part1 python=3.10.15
conda activate Rap_part1
pip install -r requirements_part1.txt
```

### 2. Set up the `Rap_part2` environment
This environment is exclusively used for LLM inference to avoid package conflicts.

```bash
conda create -n Rap_part2 python=3.10.15
conda activate Rap_part2
pip install -r requirements_part2.txt
```

💡 Troubleshooting: If encountering package conflicts, manually install dependencies using:
```bash
pip install <package_name>==<specific_version>
```
---
## 🚀 Usage

### Phase 0: Data Preparation

Prepare the datasets and embeddings for `data/` folder.

### Phase 1: Prompt Sanitization and Privacy Evaluation 

🔑 Activate the Rap_part1 environment and run the following command:

```bash
conda activate Rap_part1
python _run_script.py
```

🤖 The script will batch process all methods on all downstream tasks, including privacy risk identification, prompt sanitization, attack evaluation, and privacy analysis.


🏷️ To run a specific method, you can run "main_...py" or "baseline_...py" in the root directory or in the baseline folder. Don't forget to modify the args file to configure parameters (such as tasks or datasets).

🔖 You can run the files in Privacy_Analysis folder to get the privacy analysis results.


### Phase 2: LLM Inference

🔨 update the `main_blackllm.py` file by replacing the `black_llm_model_name` and `api_key` with your own values. Ensure that the `base_url` in `_04black_llm\black_llm_interface.py` is correct.

🔑 Activate the Rap_part2 environment and run the following command:

```bash
conda activate Rap_part2
python main_blackllm.py
```

🤖 It will run LLM inference for all methods on all downstream tasks.

---
📦 **Finally, you can get detailed results and brief reports with json and csv files.**
