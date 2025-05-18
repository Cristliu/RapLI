import os
from datasets import load_dataset, load_from_disk, Dataset
import pandas as pd
import json

class DataLoader:
    def __init__(self, task, define_dataset):
        self.task = task
        self.define_dataset = define_dataset

    def load_and_preprocess(self):
        """
        ag_news: text=content[1], label=int(content[0])
        samsum: text=item["dialogue"], label=item["summary"]
        Download if not exist (example only).
        """
        prepro_dataset = None
        dataset_path = os.path.join("./data", self.task, self.define_dataset)

        # Directly load local dataset
        if self.define_dataset == "ag_news":
            tsv_file = os.path.join(dataset_path, 'test.tsv') 
            # Read tsv file
            if os.path.exists(tsv_file):
                with open(tsv_file, 'r', encoding='utf-8') as f:
                    next(f)  # Skip the header
                    data = [line.strip().split('\t') for line in f]

                    prepro_dataset = Dataset.from_dict({
                        "text": [x[1] for x in data],
                        "label": [int(x[0]) for x in data]
                    })

                    # #Only test first 100 lines
                    # prepro_dataset = prepro_dataset.select(range(100))

        elif self.define_dataset == "piidocs":
            tsv_file = os.path.join(dataset_path, 'piidocs.tsv')
            # Read tsv file
            if os.path.exists(tsv_file):
                with open(tsv_file, 'r', encoding='utf-8') as f:
                    next(f)  # Skip the header
                    data = [line.strip().split('\t') for line in f]

                    prepro_dataset = Dataset.from_dict({
                        "text": [x[1] for x in data],
                        "label": [int(x[0]) for x in data]
                    })

                    
        elif self.define_dataset == "samsum":
            json_file = os.path.join(dataset_path, 'samsum_combined.json')
            # Read json file
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    prepro_dataset = Dataset.from_dict({
                        "text": [item["dialogue"] for item in data],
                        "label": [item["summary"] for item in data],
                    })
        
        return prepro_dataset