import tensorflow as tf

IMAGE_SIZE = 32
IMAGE_WIDTH = 256
MIN_HEIGHT = 32
MAX_HEIGHT = 2000 

def html_to_tensor(html_string):
    byte_data = html_string.encode("utf-8")
    byte_tensor = tf.convert_to_tensor(byte_data, dtype=tf.string)
    byte_array = tf.io.decode_raw(byte_tensor, tf.uint8)
    byte_array = tf.cast(byte_array, tf.float32)

    length = tf.shape(byte_array)[0]

    # limit maximum bytes
    max_bytes = IMAGE_WIDTH * MAX_HEIGHT
    byte_array = byte_array[:max_bytes]
    length = tf.shape(byte_array)[0]

    # compute adaptive height
    height = tf.cast(tf.math.ceil(length / IMAGE_WIDTH), tf.int32)

    # pad to full rectangle
    pad_len = height * IMAGE_WIDTH - length
    byte_array = tf.pad(byte_array, [[0, pad_len]])

    image = tf.reshape(byte_array, (height, IMAGE_WIDTH, 1))

    # enforce minimum height
    pad_height = tf.maximum(0, MIN_HEIGHT - tf.shape(image)[0])
    image = tf.pad(image, [[0, pad_height], [0, 0], [0, 0]])

    return image

def html_to_fixed_tensor(html_string, size=IMAGE_SIZE):
    # convert string to byte tensor
    byte_data = html_string.encode("utf-8")
    byte_tensor = tf.convert_to_tensor(byte_data, dtype=tf.string)

    # decode to uint8
    byte_array = tf.io.decode_raw(byte_tensor, tf.uint8)
    byte_array = tf.cast(byte_array, tf.float32)

    required_len = size * size
    length = tf.shape(byte_array)[0]

    # truncate if too long
    byte_array = byte_array[:required_len]
    length = tf.shape(byte_array)[0]

    # pad if too short
    pad_len = tf.maximum(0, required_len - length)
    byte_array = tf.pad(byte_array, [[0, pad_len]])

    # reshape to fixed image
    image = tf.reshape(byte_array, (size, size, 1))

    return image