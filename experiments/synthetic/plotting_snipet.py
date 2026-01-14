import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv(
    "logs/data_size_trigonometric-linear_default_gaussian.csv"
)

# Methods to plot (paper order)
methods = [
    "logging",
    "uniform",
    "greedy",
    "regression-based (single)",
    "IS-based (single)",
    "hybrid (single)",
    "IS-based (dso)",
    "hybrid (fixed-action)",
    "online",
]

# Aggregate: mean and standard error
grouped = df.groupby("target_value")

means = grouped[methods].mean()
stderr = grouped[methods].std() / np.sqrt(grouped.size().values[:, None])
x_labels = means.index.tolist()                 # [500, 1000, 2000, 4000, 8000]
x_pos = np.arange(len(x_labels))                # [0, 1, 2, 3, 4]

plt.figure(figsize=(8, 6))

for method in methods:
    plt.plot(
        x_pos,
        means[method],
        marker="o",
        linewidth=2 if method == "IS-based (dso)" else 1.5,
        label=method,
    )
    plt.fill_between(
        x_pos,
        means[method] - stderr[method],
        means[method] + stderr[method],
        alpha=0.15,
    )

plt.xticks(x_pos, x_labels)
plt.xlabel("Number of samples")
plt.ylabel("Policy value")
plt.title("Synthetic experiment (data size)")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.show()
