import tensorflow as tf
import pandas as pd
from htmltotensor import html_to_tensor

# Optimized buckets based on your cumulative distribution
# Most images are below 20,000 height, so we focus buckets there
# Using more granular buckets for common sizes, fewer for rare large sizes
BUCKETS = [42, 200, 508, 1075, 1707, 3764, 8740]

# Corresponding batch sizes - smaller batches for larger images
# This prevents GPU OOM while maintaining training efficiency
BATCHES = [128, 64, 32, 16, 8, 4, 2, 1]


def load_dataset(split="train", use_bucketing=True):
    """
    Load dataset with optional bucketing for variable-sized images.
    
    Args:
        split: 'train', 'val', or 'test'
        use_bucketing: If False, uses fixed batching (useful for debugging)
    """
    df = pd.read_csv(f"data/{split}.csv")
    
    paths = df["path"].values
    labels = df["label"].values
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    def _load(path, label):
        img = html_to_tensor(path)
        return img, label
    
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle only for training
    if split == "train":
        ds = ds.shuffle(1000)
    
    if use_bucketing:
        # Use bucketing with padding disabled to save memory
        ds = ds.apply(
            tf.data.experimental.bucket_by_sequence_length(
                element_length_func=lambda img, lbl: tf.shape(img)[0],
                bucket_boundaries=BUCKETS,
                bucket_batch_sizes=BATCHES,
                # CRITICAL: Set to False to avoid padding small images to large sizes
                pad_to_bucket_boundary=False,
                # Pad only within the batch to the max size in that batch
                padded_shapes=([None, None, None], [])
            )
        )
    else:
        # Simple batching for debugging
        ds = ds.padded_batch(
            BATCHES[0],
            padded_shapes=([None, None, None], []),
            drop_remainder=False
        )
    
    return ds.prefetch(tf.data.AUTOTUNE)


def load_dataset_with_resize(split="train", target_height=2048, batch_size=32):
    """
    Alternative approach: Resize all images to fixed height.
    This uses more compute but consistent memory.
    
    Args:
        split: 'train', 'val', or 'test'
        target_height: Height to resize all images to
        batch_size: Fixed batch size
    """
    df = pd.read_csv(f"data/{split}.csv")
    
    paths = df["path"].values
    labels = df["label"].values
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    def _load_and_resize(path, label):
        img = html_to_tensor(path)
        
        # Get original dimensions
        current_height = tf.shape(img)[0]
        width = tf.shape(img)[1]
        channels = tf.shape(img)[2]
        
        # Resize to target height, keeping width constant
        img_resized = tf.image.resize(
            img,
            [target_height, width],
            method=tf.image.ResizeMethod.BILINEAR,
            preserve_aspect_ratio=False
        )
        
        return img_resized, label
    
    ds = ds.map(_load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    
    if split == "train":
        ds = ds.shuffle(1000)
    
    ds = ds.batch(batch_size, drop_remainder=False)
    
    return ds.prefetch(tf.data.AUTOTUNE)
