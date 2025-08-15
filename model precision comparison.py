import numpy as np
import matplotlib.pyplot as plt


models = ["Dual Branch", "CMFD", "Error-Analysis", "Proposed Model"]

precision_values = [94.8, 72.3, 95.6, 96.0]


colors = ['skyblue', 'lightgreen', 'lightcoral', 'plum']

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(models))
bars = ax.bar(x, precision_values, color=colors, alpha=0.7, edgecolor='black')


ax.set_xlabel("Models", fontsize=12)
ax.set_ylabel("Precision (%)", fontsize=12)
ax.set_title("Model Precision Comparison", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, fontsize=10)


for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}%', ha='center', va='bottom', fontsize=10)


plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.ylim(0, 100)
plt.show()
