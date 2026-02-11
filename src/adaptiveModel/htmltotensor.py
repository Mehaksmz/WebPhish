import tensorflow as tf

IMAGE_WIDTH = 256
MIN_HEIGHT = 32
MAX_HEIGHT = 2000   # ← memory safety cap (you suggested this ✔)


def html_to_tensor(path):
    byte_data = tf.io.read_file(path)

    byte_array = tf.io.decode_raw(byte_data, tf.uint8)
    byte_array = tf.cast(byte_array, tf.float32)

    length = tf.shape(byte_array)[0]

    # -------------------------------------------------
    # 🔹 LIMIT maximum bytes BEFORE reshape (memory fix)
    # -------------------------------------------------
    max_bytes = IMAGE_WIDTH * MAX_HEIGHT
    byte_array = byte_array[:max_bytes]

    length = tf.shape(byte_array)[0]

    # ---- compute adaptive height ----
    height = tf.cast(tf.math.ceil(length / IMAGE_WIDTH), tf.int32)

    # ---- pad to full rectangle ----
    pad_len = height * IMAGE_WIDTH - length
    byte_array = tf.pad(byte_array, [[0, pad_len]])

    image = tf.reshape(byte_array, (height, IMAGE_WIDTH, 1))

    # -------------------------------------------------
    # 🔹 enforce minimum height (CNN stability)
    # -------------------------------------------------
    pad_height = tf.maximum(0, MIN_HEIGHT - tf.shape(image)[0])

    image = tf.pad(image, [[0, pad_height], [0, 0], [0, 0]])

    return image
