import pandas as pd
import matplotlib.pyplot as plt
import os


# Paths
input_file = "results/layer_perturbation_results.csv"
output_file = "figures/accuracy_vs_perturbation.png"


# Load results
df = pd.read_csv(input_file)


# Create figures folder if not exists
os.makedirs("figures", exist_ok=True)


# Plot
plt.figure(figsize=(8, 5))

for model in df["Model"].unique():
    model_data = df[df["Model"] == model]

    accuracy_by_perturbation = (
        model_data.groupby("Perturbation Percentage")["Accuracy"]
        .mean()
    )

    plt.plot(
        accuracy_by_perturbation.index,
        accuracy_by_perturbation.values,
        marker="o",
        label=model
    )


# Labels
plt.xlabel("Perturbation Percentage (%)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Weight Perturbation Percentage")
plt.legend()
plt.grid(True)


# Save figure
plt.savefig(output_file, bbox_inches="tight")
plt.close()


print(f"Saved: {output_file}")