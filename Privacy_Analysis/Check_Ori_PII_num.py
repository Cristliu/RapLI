import pandas as pd
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from _01pii_detection.combined_detector import CombinedPIIDetector


def check_pii_num(orig_text, detector):
    orig_pii = detector.detect_pii(orig_text)
    orig_pii_words = set()
    for entity in orig_pii:
        orig_pii_words.update(entity['text'].split())
    len_orig = len(orig_pii_words)
    return len_orig


'''
Dataset: ag_news
Total PII count: 66580
Average PII count per document: 8.760526315789473


Dataset: piidocs
Total PII count: 30362
Average PII count per document: 7.371206603544549
'''

def main():

    detector = CombinedPIIDetector()

    files = {
        "ag_news": "data/topic_classification/ag_news/test.tsv",
        "piidocs": "data/piidocs_classification/piidocs/piidocs.tsv"
    }

    for dataset, file_path in files.items():
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            continue

        df = pd.read_csv(file_path, sep='\t')
        
        pii_counts = df['text'].apply(lambda x: check_pii_num(x, detector))

        print(f"Dataset: {dataset}")
        print(f"Total PII count: {pii_counts.sum()}")
        print(f"Average PII count per document: {pii_counts.mean()}\n")


if __name__ == "__main__":
    main()
    