from datasets import load_dataset
import json
import os
import argparse
import csv

def download_samsum(output_dir="./data/samsum"):
    """
    Download SAMSum dataset train, validation and test subsets,
    merge validation and test into one file, and save train set separately
    """
    print("Downloading SAMSum dataset from Hugging Face...")
    
    # Load dataset, including train, validation and test parts
    dataset = load_dataset("samsum", split=["train", "validation", "test"])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge validation and test records
    val_test_records = []
    
    # Save training set separately
    train_records = []
    
    # Iterate and collect each subset
    for i, split_name in enumerate(["train", "validation", "test"]):
        data = dataset[i]
        
        # Convert data to list format
        for item in data:
            record = {
                "id": item["id"],
                "dialogue": item["dialogue"],
                "summary": item["summary"],
                "split": split_name
            }
            
            if split_name == "train":
                train_records.append(record)
            else:
                val_test_records.append(record)
    
    # Save training set as separate JSON file
    train_json_path = os.path.join(output_dir, "samsum_train.json")
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2, ensure_ascii=False)
        
    # Save training set as TSV file
    train_tsv_path = os.path.join(output_dir, "samsum_train.tsv")
    with open(train_tsv_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # Write header
        writer.writerow(["dialogue", "summary"])
        # Write data
        for record in train_records:
            writer.writerow([record["dialogue"], record["summary"]])
    
    # Save merged validation and test sets as JSON file
    json_output_path = os.path.join(output_dir, "samsum_combined.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(val_test_records, f, indent=2, ensure_ascii=False)
    
    # Save merged validation and test sets as TSV file
    tsv_output_path = os.path.join(output_dir, "samsum_combined.tsv")
    with open(tsv_output_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # Write header
        writer.writerow(["dialogue", "summary"])
        # Write data
        for record in val_test_records:
            writer.writerow([record["dialogue"], record["summary"]])
    
    print(f"Dataset saved to:")
    print(f"  - Train JSON: {train_json_path} ({len(train_records)} records)")
    print(f"  - Train TSV: {train_tsv_path} ({len(train_records)} records)")
    print(f"  - Validation and Test JSON: {json_output_path} ({len(val_test_records)} records)")
    print(f"  - Validation and Test TSV: {tsv_output_path} ({len(val_test_records)} records)")
    
    # Print examples
    print("\nTraining set examples:")
    for i in range(min(2, len(train_records))):
        print(f"ID: {train_records[i]['id']}")
        print(f"Dialogue:\n{train_records[i]['dialogue'][:300]}..." if len(train_records[i]['dialogue']) > 300 else train_records[i]['dialogue'])
        print(f"Summary: {train_records[i]['summary']}")
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SAMSum dataset")
    parser.add_argument("--output_dir", type=str, default="./data/samsum", help="Output directory")
    args = parser.parse_args()
    
    download_samsum(args.output_dir)
    print("Download complete!")