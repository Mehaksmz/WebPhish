import numpy as np
import os
from collections import Counter

FIXED_WIDTH = 256
root_dir = "/home/mehak/n96ncsr5g4-1/dataset"

def collect_heights(root_dir, fixed_width=256):
    heights = []

    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith(".html"):
                path = os.path.join(root, fname)
                with open(path, "rb") as f:
                    n_bytes = len(f.read())
                height = int(np.ceil(n_bytes / fixed_width))
                heights.append(height)

    return heights

heights = collect_heights(root_dir)

counts = Counter(heights)
total = len(heights)


items = sorted(counts.items())

cumulative = []
running = 0

for h, c in items:
    pct = c / total * 100
    running += pct
    cumulative.append((h, running))

for h, pct in cumulative:
    print(f"Height ≤ {h}: {pct:.1f}%")