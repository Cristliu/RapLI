import pandas as pd
import os
import sys
import json
import jsonlines

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from _01pii_detection.combined_detector import CombinedPIIDetector


def check_pii_num(orig_text, detector):
    """
    Check PII content.
    """
    orig_pii = detector.detect_pii(orig_text)
    orig_pii_words = set()
    for entity in orig_pii:
        orig_pii_words.update(entity['text'].split())
    len_orig = len(orig_pii_words)
    return len_orig


def check_synthpai_pii(file_path, detector):
    """
    Check PII content in synthpai dataset.
    For synthpai dataset, need to combine all comment texts for each user.
    """
    total_pii_count = 0
    user_count = 0
    total_comments = 0
    valid_comments = 0
    
    print(f"Processing SynthPAI dataset...")
    
    with jsonlines.open(file_path) as reader:
        for user_data in reader:
            user_count += 1
            user_comments = []
            
            if "comments" in user_data:
                # Record total original comments
                total_comments += len(user_data["comments"])
                
                # Filter valid comments (with text field)
                valid_user_comments = []
                for comment in user_data["comments"]:
                    if "text" in comment and comment["text"]:  # Ensure text field exists and is not empty
                        valid_user_comments.append(comment["text"])
                        valid_comments += 1
                
                # Combine valid comment texts
                if valid_user_comments:
                    all_comments = " ".join(valid_user_comments)
                    pii_count = check_pii_num(all_comments, detector)
                    total_pii_count += pii_count
    
    print(f"Dataset: synthpai")
    print(f"Total users: {user_count}")
    print(f"Total original comments: {total_comments}")
    print(f"Valid comments: {valid_comments}")
    print(f"Total PII count: {total_pii_count}")
    print(f"Average PII per user: {total_pii_count / user_count if user_count > 0 else 0}")
    print(f"Average PII per valid comment: {total_pii_count / valid_comments if valid_comments > 0 else 0}\n")
    
    return total_pii_count, user_count, valid_comments


def check_samsum_pii(file_path, detector):
    """
    Check PII content in samsum dataset.
    For samsum dataset, need to check PII in the dialogue field.
    """
    print(f"Processing SAMSum dataset...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check data integrity
    valid_records = []
    for item in data:
        if "dialogue" in item and "summary" in item:
            valid_records.append(item)
    
    if not valid_records:
        print(f"Warning: No valid records containing dialogue and summary fields in {file_path}!")
        return 0, 0
    
    # Calculate PII
    pii_counts = [check_pii_num(item["dialogue"], detector) for item in valid_records]
    total_pii = sum(pii_counts)
    
    print(f"Dataset: samsum")
    print(f"Valid dialogue records: {len(valid_records)}")
    print(f"Total PII count: {total_pii}")
    print(f"Average PII per dialogue: {total_pii / len(valid_records) if valid_records else 0}\n")
    
    return total_pii, len(valid_records)



def main():
    detector = CombinedPIIDetector()

    # Process three new datasets
    files = {
        "synthpai": "data/synthpai/comments_eval_revised.jsonl",
        "samsum": "data/samsum/samsum/samsum_combined.json",
    }
    
    summary = []
    
    # Check synthpai dataset
    synthpai_path = files["synthpai"]
    if os.path.exists(synthpai_path):
        total_pii, user_count, comment_count = check_synthpai_pii(synthpai_path, detector)
        summary.append({
            "dataset": "synthpai",
            "records": user_count,
            "comments/dialogues/stories": comment_count,
            "total_pii": total_pii,
            "avg_pii_per_record": total_pii / user_count if user_count > 0 else 0,
            "avg_pii_per_comment": total_pii / comment_count if comment_count > 0 else 0
        })
    else:
        print(f"File {synthpai_path} does not exist.")
    
    # Check samsum dataset
    samsum_path = files["samsum"]
    if os.path.exists(samsum_path):
        total_pii, record_count = check_samsum_pii(samsum_path, detector)
        summary.append({
            "dataset": "samsum",
            "records": record_count,
            "total_pii": total_pii,
            "avg_pii_per_record": total_pii / record_count if record_count > 0 else 0
        })
    else:
        print(f"File {samsum_path} does not exist.")
    
    
    # Print summary information
    print("\n=== Dataset PII Content Summary ===")
    for item in summary:
        print(f"Dataset: {item['dataset']}")
        print(f"Record count: {item['records']}")
        if 'comments/dialogues/stories' in item:
            print(f"Total comments: {item['comments/dialogues/stories']}")
        print(f"Total PII count: {item['total_pii']}")
        print(f"Average PII per record: {item['avg_pii_per_record']:.4f}")
        if 'avg_pii_per_comment' in item:
            print(f"Average PII per comment: {item['avg_pii_per_comment']:.4f}")
        print("-" * 40)
    

'''
Dataset: synthpai
Total users: 300
Total original comments: 7785
Valid comments: 7785
Total PII count: 2688
Average PII per user: 8.96
Average PII per valid comment: 0.34527938342967246

Processing SAMSum dataset...
Dataset: samsum
Valid dialogue records: 1637
Total PII count: 11316
Average PII per dialogue: 6.912645082467929


=== Dataset PII Content Summary ===
Dataset: synthpai
Record count: 300
Total comments: 7785
Total PII count: 2688
Average PII per record: 8.9600
Average PII per comment: 0.3453
----------------------------------------
Dataset: samsum
Record count: 1637
Total PII count: 11316
Average PII per record: 6.9126
----------------------------------------
'''

if __name__ == "__main__":
    main()