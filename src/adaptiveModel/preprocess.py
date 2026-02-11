import tensorflow as tf
import pandas as pd
from htmltotensor import html_to_tensor

def load_dataset(split="train"):

    df = pd.read_csv(f"data/{split}.csv")

    paths = df["path"].values
    labels = df["label"].values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img = html_to_tensor(path)
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if split == "train":
        ds = ds.shuffle(1000)

    # ❌ remove bucketing
    # ❌ remove batching

    return ds.prefetch(tf.data.AUTOTUNE)
