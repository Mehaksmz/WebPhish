import pandas as pd
from sklearn.model_selection import train_test_split
import os

INPUT_CSV = "data/metadata.csv"
OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load full dataset metadata
df = pd.read_csv(INPUT_CSV)

# --- First split: Train (70%) and Temp (30%) ---
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

# --- Second split: Val (15%) and Test (15%) ---
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# --- Save splits ---
train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

print("Stratified CSV splits created:")
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))
