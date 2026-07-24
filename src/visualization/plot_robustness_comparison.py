import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_robustness_comparison():

    input_file = "results/analysis/robustness_analysis_results.csv"
    output_file = "figures/robustness_comparison.png"

    os.makedirs("figures", exist_ok=True)

    # Load robustness results
    df = pd.read_csv(input_file)

    # Extract average robustness scores
    comparison = (
        df.groupby("Model")["Average_Robustness_Score"]
        .first()
        .reset_index()
    )

    # Plot
    plt.figure(figsize=(8, 5))

    plt.bar(
        comparison["Model"],
        comparison["Average_Robustness_Score"]
    )

    plt.xlabel("Model")
    plt.ylabel("Average_Robustness_Score")
    plt.title("Model Robustness Comparison")
    plt.grid(axis="y")

    # Save figure
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")


if __name__ == "__main__":
    plot_robustness_comparison()