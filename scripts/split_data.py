import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/HAM10000_metadata.csv")

malignant = {'mel', 'bcc', 'akiec'}
df['label'] = df['dx'].apply(lambda x: 1 if x in malignant else 0)

# get unique lesion_ids, each tagged with lesion's own label
# (assume all images of same lesion share same dx - true for this dataset)
lesion_labels = df.groupby('lesion_id')['label'].first().reset_index()

# split lesion_ids: 80% train, 10% val, 10% test — stratify by label to keep ratio
train_ids, temp_ids = train_test_split(
    lesion_labels, test_size=0.2, stratify=lesion_labels['label'], random_state=42
)
val_ids, test_ids = train_test_split(
    temp_ids, test_size=0.5, stratify=temp_ids['label'], random_state=42
)

train_df = df[df['lesion_id'].isin(train_ids['lesion_id'])]
val_df   = df[df['lesion_id'].isin(val_ids['lesion_id'])]
test_df  = df[df['lesion_id'].isin(test_ids['lesion_id'])]

print("Train images:", len(train_df), "| lesions:", train_df['lesion_id'].nunique())
print("Val images:  ", len(val_df),   "| lesions:", val_df['lesion_id'].nunique())
print("Test images: ", len(test_df),  "| lesions:", test_df['lesion_id'].nunique())

print("\nTrain label dist:\n", train_df['label'].value_counts())
print("\nVal label dist:\n", val_df['label'].value_counts())
print("\nTest label dist:\n", test_df['label'].value_counts())

# save splits
train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)
print("\nSaved to data/processed/")