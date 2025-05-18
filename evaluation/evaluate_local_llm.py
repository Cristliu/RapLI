import json
import os
from sklearn.metrics import accuracy_score, f1_score

def evaluate_results_llm(task, define_dataset, csv_file_name, ground_truths, predictions, local_llm_model, avg_time):
    results = {}  
    if task in ["topic_classification", "piidocs_classification"]:
        accuracy = accuracy_score(ground_truths, predictions)
        f1 = f1_score(ground_truths, predictions, average='weighted')
        results["accuracy"] = accuracy
        results["f1_score"] = f1
        results["avg_time"] = avg_time

    print(f"Evaluation Results for {task}_{define_dataset}_llama8b/{csv_file_name}:")
    for metric, value in results.items():
        print(f"{metric}: {value}") 


    save_path = os.path.join("evaluation_results", f"{task}_{define_dataset}_gpt-3.5-turbo-ca_llama8b_denoise","Local_Eval_Results")
    os.makedirs(save_path, exist_ok=True)

    json_filename = f"LocalEvalResults_{csv_file_name}.json"
    json_save_path = os.path.join(save_path, json_filename)

    try:
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {json_save_path}\n\n")
    except Exception as e:
        print(f"Error saving JSON {json_save_path}: {e}")
