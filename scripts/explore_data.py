import pandas as pd

# load metadata
df = pd.read_csv("data/raw/HAM10000_metadata.csv")

print("Total rows:", len(df))
print("\nColumns:", df.columns.tolist())
print("\nDiagnosis counts:\n", df['dx'].value_counts())
print("\nUnique lesion_id count:", df['lesion_id'].nunique())

# binary mapping
malignant = {'mel', 'bcc', 'akiec'}
df['label'] = df['dx'].apply(lambda x: 1 if x in malignant else 0)  # 1=malignant, 0=benign

print("\nBinary label counts:\n", df['label'].value_counts())
print("\nSample rows:\n", df[['image_id', 'dx', 'label', 'lesion_id']].head())