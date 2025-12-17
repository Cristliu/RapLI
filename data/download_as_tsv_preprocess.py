import os
from datasets import load_dataset
import pandas as pd

# Download the dataset
dataset = load_dataset("gretelai/gretel-pii-masking-en-v1")#maxmoynan/SemEval2017-Task4aEnglish 

# Load train, validation, and test data, and keep data that meets the following filtering criteria
train_data = dataset['train']
validation_data = dataset['validation']
test_data = dataset['test']

# From the domain column, keep only data equal to ['healthcare', 'healthcare-administration', 'legal-documents', 'travel-hospitality'], and for travel-hospitality, only use certain document_types to avoid confusion: isin FeedbackForm BaggagePolicy Itinerary E-Ticket HotelReservation.
train_df = pd.DataFrame(train_data)
train_df = train_df[train_df['domain'].isin(['healthcare', 'healthcare-administration', 'legal-documents', 'travel-hospitality'])]
train_df = train_df[~((train_df['domain']=='travel-hospitality') & (~train_df['document_type'].isin(['FeedbackForm', 'BaggagePolicy', 'Itinerary', 'E-Ticket', 'HotelReservation'])))]

val_df = pd.DataFrame(validation_data)
val_df = val_df[val_df['domain'].isin(['healthcare', 'healthcare-administration', 'legal-documents', 'travel-hospitality'])]
val_df = val_df[~((val_df['domain']=='travel-hospitality') & (~val_df['document_type'].isin(['FeedbackForm', 'BaggagePolicy', 'Itinerary', 'E-Ticket', 'HotelReservation'])))]

# Convert to Pandas DataFrame
test_df = pd.DataFrame(test_data)
test_df = test_df[test_df['domain'].isin(['healthcare', 'healthcare-administration', 'legal-documents', 'travel-hospitality'])]
# When domain='travel-hospitality', keep only data with document_type in FeedbackForm BaggagePolicy Itinerary E-Ticket HotelReservation, other domain categories remain unchanged
test_df = test_df[~((test_df['domain']=='travel-hospitality') & (~test_df['document_type'].isin(['FeedbackForm', 'BaggagePolicy', 'Itinerary', 'E-Ticket', 'HotelReservation'])))]

# Use a loop to complete the above work and concatenate all data together to form a new df_combined
df_combined = pd.DataFrame()
for data in [train_data, validation_data, test_data]:
    df = pd.DataFrame(data)
    df = df[df['domain'].isin(['healthcare', 'healthcare-administration', 'legal-documents', 'travel-hospitality'])]
    df = df[~((df['domain']=='travel-hospitality') & (~df['document_type'].isin(['FeedbackForm', 'BaggagePolicy', 'Itinerary', 'E-Ticket', 'HotelReservation'])))]

    df_combined = pd.concat([df_combined, df])

df_combined = df_combined.reset_index(drop=True)

# Not using all categories because this dataset is not constructed for classification, but for PII leakage detection
# 'healthcare', 'healthcare-administration', indicate which document_types are included when constructing the prompt
# 'legal-documents', indicate which document_types are included when constructing the prompt
# 'travel-hospitality'-- we only use certain document_types because other categories may cause confusion: isin FeedbackForm BaggagePolicy Itinerary E-Ticket HotelReservation
 

print(f"Dataset size: {df_combined.shape}")
print(f"Dataset columns: {df_combined.columns}")
print(f"First 5 rows of the dataset: {df_combined.head()}")

# Keep domain renamed as label, corresponding relationship is healthcare:0, healthcare-administration:0, legal-documents:1, travel-hospitality:2; keep text as text, delete other columns
df_combined = df_combined.rename(columns={'domain': 'label', 'text': 'text'})
df_combined['label'] = df_combined['label'].map({'healthcare': 0, 'healthcare-administration': 0, 'legal-documents': 1, 'travel-hospitality': 2})
df_combined = df_combined[['label', 'text']]
# Save as TSV file
path = "./data/piidocs_classification/piidocs/"
# Ensure the directory exists
if not os.path.exists(path):
    os.makedirs(path)

# Preprocess the dataset, remove newline characters, ensure each line has only one text (string) and one label (integer)
# Write the above many lines into one line
# df_combined['text'] = df_combined['text'].str.replace('\n|\r|\t|\v|\f|\u2028|\u2029|\u200b|\u200c|\u200d|\u200e|\u200f|\u202a|\u202b|\u202c|\u202d|\u202e', ' ')
df_combined['text'] = df_combined['text'].astype(str) \
    .replace(r'[\r\n\t]+', ' ', regex=True)          # Replace all newline, carriage return, and tab characters with spaces
df_combined['text'] = df_combined['text'].str.replace(r'\s+', ' ', regex=True)  # Merge consecutive spaces into one
df_combined['text'] = df_combined['text'].str.strip('" ')

print(f"Dataset size: {df_combined.shape}")
print(f"Dataset columns: {df_combined.columns}")
print(f"First 5 rows of the dataset: {df_combined.head()}")

df_combined.to_csv(path+"piidocs.tsv", sep='\t', index=False)

# Map the values of the first column to numbers, corresponding relationship is neutral:0, entailment:1, contradiction:2
# test_df['gold_label'] = test_df['gold_label'].map({'neutral': 0, 'entailment': 1, 'contradiction': 2})

# Save as TSV file
# test_df.to_csv("data/clinical_inference/mednli/mednli.tsv", sep='\t', index=False)

# Output the saved file path
# print("Test data saved as 'mednli.tsv'")
