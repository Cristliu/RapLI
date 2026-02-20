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

### 3. Note on Specific Folders

💡 **Note**: When running code in specific subfolders (e.g., `interactive_demo`), please check their respective `README.md` and `requirements.txt` for additional environment requirements.

💡 **Troubleshooting**: If encountering package conflicts, manually install dependencies using:

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
🔖 You can run the files in `Privacy_Analysis` folder to get the privacy analysis results.

🔖 You can run the files in `intermediate_metrics_Analysis` folder to get the Intermediate Metrics that measure text quality after sanitization.

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

---

## 💻 Interactive Demo: Privacy-Preserving LLM Chat System

To validate the practical applicability and usability of Rap-LI, we designed a **Privacy-Preserving LLM Chat System** (similar to ChatGPT or Gemini) that integrates the proposed Rap-LI with an interactive interface.


### Key Features

* **Interactive Risk Adjustment**: Users can manually adjust token risk levels via an interactive pop-up window. The system automatically marks privacy entities based on Rap-LI's defaults, which users can then fine-tune.
* **Side-by-Side Comparison**: To enhance user experience, the system displays the original prompt and the sanitized prompt side-by-side. Replaced entities are highlighted using colors corresponding to their risk levels.
* **One-Click Restoration**: The system provides a functional module to extract and restore original private entities from the LLM output, improving readability and task completion.
* **Customizable Sensitivity Database**: Supports industry- or user-specific sensitivity databases. Custom rules are automatically mapped to risk levels to reduce manual adjustments.

### Resources

* **Source Code**: The complete system code is available in the `interactive_demo/` folder. Please refer to `interactive_demo/README.md` for detailed setup and usage instructions.
* **Video Tutorial & Demo**: You can watch the system introduction and usage tutorial videos here: [Google Drive Link](https://drive.google.com/drive/folders/1i4T3mNYpwI6BQmr5cRNuAYJpxR0fen_D?usp=drive_link)

---
