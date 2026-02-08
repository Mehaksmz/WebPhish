import os
import pandas as pd
import numpy as np

DATASET_ROOT = "data/n96ncsr5g4-1/n96ncsr5g4-1/dataset"
LABEL_CSV = "data/index2.csv"
OUTPUT_META = "data/metadata.csv"

IMAGE_WIDTH = 256   

df = pd.read_csv(LABEL_CSV)
label_map = dict(zip(df["filename"], df["label"]))

rows = []

for root, _, files in os.walk(DATASET_ROOT):
    for f in files:
        if not f.endswith(".html"):
            continue

        if f not in label_map:
            continue

        path = os.path.join(root, f)

        with open(path, "rb") as file:
            byte_len = len(file.read())

        height = int(np.ceil(byte_len / IMAGE_WIDTH))

        rows.append({
            "path": path,
            "label": label_map[f],
            "height": height
        })

pd.DataFrame(rows).to_csv(OUTPUT_META, index=False)

print("Metadata created:", OUTPUT_META)
