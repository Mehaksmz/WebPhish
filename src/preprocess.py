import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil

BASE_IMG_DIR = "/home/mehak/radomised_image_dataset"
OUTPUT_DIR = "/home/mehak/radomised_image_dataset_split"


df = pd.read_csv("data/index.csv")  

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

def build_label_map(df):
    return dict(zip(df["filename"], df["label"]))

          
def copy_images(base_dir, split_df, split_name, output_dir):
    label_map = build_label_map(split_df)

    for part_name in os.listdir(base_dir):
        part_path = os.path.join(base_dir, part_name)

        if not os.path.isdir(part_path):
            continue

        print(f"Processing {split_name}: {part_name}")

        for root, _, files in os.walk(part_path):
            for filename in files:
                if not filename.lower().endswith(".png"):
                    continue

                if filename not in label_map:
                    continue  

                label = "phishing" if label_map[filename] == 1 else "legitimate"

                src = os.path.join(root, filename)
                dst_dir = os.path.join(output_dir, split_name, label)
                os.makedirs(dst_dir, exist_ok=True)

                shutil.copy(src, os.path.join(dst_dir, filename))

copy_images(BASE_IMG_DIR, train_df, "train", OUTPUT_DIR)
copy_images(BASE_IMG_DIR, val_df, "val", OUTPUT_DIR)
copy_images(BASE_IMG_DIR, test_df, "test", OUTPUT_DIR)

