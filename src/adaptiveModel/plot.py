import re
import matplotlib.pyplot as plt

heights = []
percentages = []


with open("cumulitive.txt", "r", encoding="utf-8") as f:
    for line in f:
        # Match lines like: Height ≤ 10: 11.0%
        match = re.search(r'Height\s≤\s(\d+):\s*([\d.]+)%', line)
        if match:
            height = int(match.group(1))
            percent = float(match.group(2))
            heights.append(height)
            percentages.append(percent)


plt.figure()
plt.plot(heights, percentages, marker='o')
plt.xlabel("Height")
plt.ylabel("Cumulative Percentage")
plt.title("Cumulative Height Distribution")
plt.grid(True)

plt.show()