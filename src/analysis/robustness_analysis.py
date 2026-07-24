import pandas as pd
import os


# File paths
BASELINE_RESNET_PATH = "results/baseline_resnet50.txt"
BASELINE_INCEPTION_PATH = "results/baseline_inceptionv3.txt"

PERTURBATION_RESULTS_PATH = "results/perturbation_results.csv"
LAYER_RESULTS_PATH = "results/layer_perturbation_results.csv"
ROBUSTNESS_RESULTS_PATH = "results/analysis/robustness_analysis_results.csv"


def load_baseline_accuracy(file_path):
    """
    Load baseline accuracy from txt file.
    """

    with open(file_path, "r") as file:
        for line in file:
            if "Accuracy" in line:
                accuracy = float(line.split(":")[1].strip().replace("%", ""))
                return accuracy

    return None


def load_results():
    """
    Load baseline and perturbation results.
    """

    resnet_accuracy = load_baseline_accuracy(
        BASELINE_RESNET_PATH
    )

    inception_accuracy = load_baseline_accuracy(
        BASELINE_INCEPTION_PATH
    )


    print("ResNet-50 Baseline Accuracy:", resnet_accuracy)
    print("InceptionV3 Baseline Accuracy:", inception_accuracy)


    if os.path.exists(PERTURBATION_RESULTS_PATH):
        perturbation_results = pd.read_csv(
            PERTURBATION_RESULTS_PATH
        )

        print("\nPerturbation Results Loaded")
        print(perturbation_results.head())


    if os.path.exists(LAYER_RESULTS_PATH):
        layer_results = pd.read_csv(
            LAYER_RESULTS_PATH
        )

        print("\nLayer Perturbation Results Loaded")
        print(layer_results.head())

def calculate_accuracy_drop():
    """
    Calculate accuracy drop after weight perturbation.
    """

    # Load baseline accuracies
    resnet_baseline = load_baseline_accuracy(
        BASELINE_RESNET_PATH
    )

    inception_baseline = load_baseline_accuracy(
        BASELINE_INCEPTION_PATH
    )


    # Load layer perturbation results
    layer_results = pd.read_csv(
        LAYER_RESULTS_PATH
    )


    accuracy_drop_results = []


    for index, row in layer_results.iterrows():

        model = row["Model"]
        perturbed_accuracy = row["Accuracy"]


        if model == "ResNet-50":
            baseline_accuracy = resnet_baseline

        elif model == "InceptionV3":
            baseline_accuracy = inception_baseline

        else:
            continue


        accuracy_drop = (
            baseline_accuracy - perturbed_accuracy
        )


        accuracy_drop_results.append({
            "Model": model,
            "Region": row["Region"],
            "Perturbation_Level": row["Perturbation Percentage"],
            "Baseline_Accuracy": baseline_accuracy,
            "Perturbed_Accuracy": perturbed_accuracy,
            "Accuracy_Drop": accuracy_drop
        })


    accuracy_drop_df = pd.DataFrame(
        accuracy_drop_results
    )


    return accuracy_drop_df

def compare_model_accuracy_drop(accuracy_drop_df):
    """
    Compare accuracy drop between ResNet-50 and InceptionV3.
    """

    comparison_df = accuracy_drop_df[
        [
            "Model",
            "Region",
            "Perturbation_Level",
            "Accuracy_Drop"
        ]
    ]
    comparison_df = comparison_df.sort_values(
        by=[
            "Region",
            "Perturbation_Level",
            "Model"
        ]
    )

    return comparison_df

def calculate_robustness_score(accuracy_drop_df):
    """
    Calculate robustness score.

    Formula:
    Robustness Score = 1 - (Accuracy Drop / Baseline Accuracy)
    """

    robustness_df = accuracy_drop_df.copy()


    robustness_df["Robustness_Score"] = (
        1 -
        (
            robustness_df["Accuracy_Drop"]
            /
            robustness_df["Baseline_Accuracy"]
        )
    )


    return robustness_df

def rank_models(robustness_df):
    """
    Calculate average robustness score
    and rank models.
    """

    ranking_df = (
        robustness_df
        .groupby("Model")["Robustness_Score"]
        .mean()
        .reset_index()
    )


    ranking_df.rename(
        columns={
            "Robustness_Score": "Average_Robustness_Score"
        },
        inplace=True
    )


    ranking_df = ranking_df.sort_values(
        by="Average_Robustness_Score",
        ascending=False
    )


    ranking_df.insert(
        0,
        "Rank",
        range(1, len(ranking_df) + 1)
    )


    return ranking_df

if __name__ == "__main__":

    load_results()

    accuracy_drop_df = calculate_accuracy_drop()

    print("\nAccuracy Drop Results:")
    print(accuracy_drop_df)

    comparison_df = compare_model_accuracy_drop(
        accuracy_drop_df
    )

    print("\nModel Accuracy Drop Comparison:")
    print(comparison_df)

    robustness_df = calculate_robustness_score(
    accuracy_drop_df
)

print("\nRobustness Score Results:")
print(
    robustness_df[
        [
            "Model",
            "Region",
            "Perturbation_Level",
            "Robustness_Score"
        ]
    ]
)

ranking_df = rank_models(
    robustness_df
)

def save_robustness_results(robustness_df, ranking_df):
    """
    Save complete robustness analysis results
    into a single CSV file.
    """

    # Add ranking information to each model row
    final_df = robustness_df.merge(
        ranking_df[
            [
                "Model",
                "Rank",
                "Average_Robustness_Score"
            ]
        ],
        on="Model",
        how="left"
    )


    # Select final columns
    final_df = final_df[
        [
            "Model",
            "Region",
            "Perturbation_Level",
            "Baseline_Accuracy",
            "Perturbed_Accuracy",
            "Accuracy_Drop",
            "Robustness_Score",
            "Average_Robustness_Score",
            "Rank"
        ]
    ]


    # Save CSV
    final_df.to_csv(
        ROBUSTNESS_RESULTS_PATH,
        index=False
    )


    print(
        "\nRobustness analysis saved successfully:"
    )

    print(
        ROBUSTNESS_RESULTS_PATH
    )

save_robustness_results(
    robustness_df,
    ranking_df
)