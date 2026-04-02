import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ["Baseline CNN", "Adaptive CNN", "ResNet50", "VGG19", "InceptionV3"]

# Convert percentages to decimal form
accuracy = [0.90, 0.91, 0.70, 0.62, 0.81]
f1_scores = [0.89, 0.91, 0.70, 0.57, 0.81]

# X positions
x = np.arange(len(models))

# Create figure
plt.figure(figsize=(8,5))

# Plot lines
plt.plot(x, accuracy, marker='o', linewidth=2, label="Accuracy")
plt.plot(x, f1_scores, marker='o', linewidth=2, label="F1 Score")

# Add value labels above points
for i in range(len(models)):
    plt.text(x[i], accuracy[i] + 0.01, f"{accuracy[i]:.2f}", 
             ha='center', fontsize=9)
    plt.text(x[i], f1_scores[i] - 0.04, f"{f1_scores[i]:.2f}", 
             ha='center', fontsize=9)

# Labels and formatting
plt.xticks(x, models, rotation=20)
plt.ylabel("Score")
plt.xlabel("Model")
plt.title("Classification Performance of All Model Architectures")
plt.ylim(0.45, 1.0)  # Adjust if needed
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()