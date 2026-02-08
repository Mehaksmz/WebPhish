import tensorflow as tf
import pandas as pd
from htmltotensor import html_to_tensor

BUCKETS = [128, 256, 512, 1024, 2048]
BATCHES = [64, 48, 32, 16, 8, 4]


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

    ds = ds.apply(
        tf.data.experimental.bucket_by_sequence_length(
            element_length_func=lambda img, lbl: tf.shape(img)[0],
            bucket_boundaries=BUCKETS,
            bucket_batch_sizes=BATCHES,
            pad_to_bucket_boundary=False
        )
    )

    return ds.prefetch(tf.data.AUTOTUNE)
