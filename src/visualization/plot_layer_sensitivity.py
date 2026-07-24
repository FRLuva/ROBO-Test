import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_layer_sensitivity():

    input_file = "results/layer_perturbation_results.csv"
    output_file = "figures/layer_sensitivity.png"

    os.makedirs("figures", exist_ok=True)

    # Load results
    df = pd.read_csv(input_file)

    # Load baseline accuracies
    with open("results/baseline_resnet50.txt") as f:
      for line in f:
        if "Top-1 Accuracy" in line:
            resnet_baseline = float(
                line.split(":")[1].replace("%", "").strip()
        )


    with open("results/baseline_inceptionv3.txt") as f:
       for line in f:
          if "Top-1 Accuracy" in line:
            inception_baseline = float(
                line.split(":")[1].replace("%", "").strip()
           )

    # Assign baseline accuracy
    df["Baseline Accuracy"] = df["Model"].apply(
        lambda x: resnet_baseline if "ResNet" in x else inception_baseline
    )

    # Calculate accuracy drop
    df["Accuracy Drop"] = (
        df["Baseline Accuracy"] - df["Accuracy"]
    )

    # Average drop by model and region
    sensitivity = (
        df.groupby(["Model", "Region"])["Accuracy Drop"]
        .mean()
        .reset_index()
    )

    # Plot
    plt.figure(figsize=(10, 6))

    for model in sensitivity["Model"].unique():

        model_data = sensitivity[
            sensitivity["Model"] == model
        ]

        plt.bar(
            model_data["Region"],
            model_data["Accuracy Drop"],
            label=model,
            alpha=0.7
        )

    plt.xlabel("Layer Region")
    plt.ylabel("Average Accuracy Drop")
    plt.title("Layer Sensitivity Analysis")
    plt.legend()
    plt.grid(axis="y")

    # Save figure
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")


if __name__ == "__main__":
    plot_layer_sensitivity()