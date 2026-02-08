import tensorflow as tf
import numpy as np

IMAGE_WIDTH = 256

def html_to_tensor(path):
    byte_data = tf.io.read_file(path)

    byte_array = tf.io.decode_raw(byte_data, tf.uint8)
    byte_array = tf.cast(byte_array, tf.float32)

    length = tf.shape(byte_array)[0]
    height = tf.cast(tf.math.ceil(length / IMAGE_WIDTH), tf.int32)

    pad_len = height * IMAGE_WIDTH - length
    byte_array = tf.pad(byte_array, [[0, pad_len]])

    image = tf.reshape(byte_array, (height, IMAGE_WIDTH, 1))

    return image