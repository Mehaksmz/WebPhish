import os
import numpy as np
from PIL import Image


DATASET_ROOT = r"/home/mehak/n96ncsr5g4-1/n96ncsr5g4-1/dataset"
OUTPUT_ROOT = r"/home/mehak/radomised_image_dataset"
IMAGE_SIZE = 256


def html_to_image(html_path, size=256):
    with open(html_path, "rb") as f:   
        byte_array = np.frombuffer(f.read(), dtype=np.uint8)

    required_len = size * size

    if len(byte_array) < required_len:
        pad_len = required_len - len(byte_array)
        random_pad = np.random.randint(0, 256, pad_len, dtype=np.uint8)
        byte_array = np.concatenate([byte_array, random_pad])
    else:
        byte_array = byte_array[:required_len]

    image_array = byte_array.reshape((size, size))
    return Image.fromarray(image_array, mode="L")


def process_dataset(dataset_root, output_root):
    for part_name in os.listdir(dataset_root):
        part_path = os.path.join(dataset_root, part_name)

        if not os.path.isdir(part_path):
            continue

        print(f"Processing folder: {part_name}")

        for root, _, files in os.walk(part_path):
            for filename in files:
                if not filename.lower().endswith(".html"):
                    continue

                html_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, dataset_root)
                output_dir = os.path.join(output_root, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                png_name = filename.replace(".html", ".png")
                output_path = os.path.join(output_dir, png_name)

                try:
                    img = html_to_image(html_path, IMAGE_SIZE)
                    img.save(output_path)
                except Exception as e:
                    print(f"Error processing {html_path}: {e}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    process_dataset(DATASET_ROOT, OUTPUT_ROOT)
    print("All HTML files converted successfully.")
