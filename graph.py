import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
models = ["Baseline CNN\n(32×32)", "Adaptive CNN", "ResNet50", "VGG19"]

TP = [7020, 7007, 4722, 6614]
FP = [480,  493,  2778,  886]
FN = [778,  627,   839, 3615]
TN = [3722, 3873,  3661,  885]

# ── Style ─────────────────────────────────────────────────────────────────────
colors = {
    "TP": "#2196F3",   # blue
    "TN": "#4CAF50",   # green
    "FP": "#FF5722",   # red-orange
    "FN": "#FF9800",   # amber
}

x = np.arange(len(models))
bar_width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Confusion Matrix Breakdown by Model", fontsize=15, fontweight="bold", y=1.02)

# ── Left chart: TP & TN (correct predictions) ─────────────────────────────────
b1 = ax1.bar(x - bar_width / 2, TP, bar_width, label="TP (True Positive)",
             color=colors["TP"], edgecolor="white", linewidth=0.8)
b2 = ax1.bar(x + bar_width / 2, TN, bar_width, label="TN (True Negative)",
             color=colors["TN"], edgecolor="white", linewidth=0.8)

# value labels
for bar in b1:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
             f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8.5)
for bar in b2:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
             f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8.5)

ax1.set_title("Correct Predictions (TP & TN)", fontsize=12, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
ax1.set_ylabel("Count", fontsize=11)
ax1.set_ylim(0, max(max(TP), max(TN)) * 1.18)
ax1.legend(fontsize=10)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# ── Right chart: FP & FN (errors) ─────────────────────────────────────────────
b3 = ax2.bar(x - bar_width / 2, FP, bar_width, label="FP (False Positive)",
             color=colors["FP"], edgecolor="white", linewidth=0.8)
b4 = ax2.bar(x + bar_width / 2, FN, bar_width, label="FN (False Negative)",
             color=colors["FN"], edgecolor="white", linewidth=0.8)

for bar in b3:
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
             f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8.5)
for bar in b4:
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
             f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8.5)

ax2.set_title("Misclassifications (FP & FN)", fontsize=12, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=10)
ax2.set_ylabel("Count", fontsize=11)
ax2.set_ylim(0, max(max(FP), max(FN)) * 1.18)
ax2.legend(fontsize=10)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("confusion_matrix_charts.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → confusion_matrix_charts.png")