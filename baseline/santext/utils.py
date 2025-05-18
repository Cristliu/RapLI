import os
import unicodedata
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

def word_normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def get_vocab_agnews(data_dir, tokenizer, tokenizer_type="subword"): 
    vocab = Counter()

    # Path to the TSV file
    tsv_file = os.path.join(data_dir, 'test.tsv') #TSV fields are label,text

    if os.path.exists(tsv_file):
        # If the TSV file exists, read it and update the vocab
        with open(tsv_file, 'r', encoding='utf-8') as f:
            next(f)  # Skip the header
            for line in tqdm(f, desc="Reading existing TSV file"):
                label, text = line.strip().split('\t')
                if tokenizer_type == "subword":
                    tokens = tokenizer.tokenize(text)
                else:
                    tokens = text.split()
                vocab.update(tokens)
    else:
        # Load dataset from Huggingface
        dataset = load_dataset('ag_news', split='test')
        
        # Save the dataset to a TSV file
        with open(tsv_file, 'w', encoding='utf-8') as f:
            f.write("label\ttext\n")
            for example in tqdm(dataset, total=len(dataset)):
                text = example['text']
                label = example['label']
                f.write(f"{label}\t{text}\n")
                
                if tokenizer_type == "subword":
                    tokens = tokenizer.tokenize(text)
                else:
                    tokens = text.split()
                vocab.update(tokens)
    
    return vocab



def get_vocab_piidocs(data_dir, tokenizer, tokenizer_type="subword"): 
    vocab = Counter()

    # Path to the TSV file
    tsv_file = os.path.join(data_dir, 'piidocs.tsv') #TSV fields are label,text

    if os.path.exists(tsv_file):
        # If the TSV file exists, read it and update the vocab
        with open(tsv_file, 'r', encoding='utf-8') as f:
            next(f)  # Skip the header
            for line in tqdm(f, desc="Reading existing TSV file"):
                label, text = line.strip().split('\t')
                if tokenizer_type == "subword":
                    tokens = tokenizer.tokenize(text)
                else:
                    tokens = text.split()
                vocab.update(tokens)
    else:
        print("The TSV file does not exist. Please run the download_as_tsv_preprocess.py script to generate the TSV file.")
    return vocab



# Add function to build vocabulary for SAMSUM dataset
def get_vocab_samsum(data_dir, tokenizer, tokenizer_type="subword"):
    vocab = Counter()
    
    # JSON file path
    json_file = os.path.join(data_dir, 'samsum_combined.json')
    
    if os.path.exists(json_file):
        # If JSON file exists, read it and update vocabulary
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in tqdm(data, desc="Processing SAMSUM dataset"):
                text = item["dialogue"]
                if tokenizer_type == "subword":
                    tokens = tokenizer.tokenize(text)
                else:
                    tokens = text.split()
                vocab.update(tokens)
    else:
        print(f"Error: SAMSUM dataset file {json_file} not found!")
    
    return vocab