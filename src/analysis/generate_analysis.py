import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


INPUT_FILE = Path(
    "results/layer_perturbation_results.csv"
)


OUTPUT_DIR = Path(
    "results/analysis"
)


def create_output_folder():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )



def generate_summary_table(df):

    output_file = (
        OUTPUT_DIR /
        "final_comparison_table.csv"
    )


    df.to_csv(
        output_file,
        index=False
    )


    print(
        "Comparison table saved:"
    )

    print(
        output_file
    )



def plot_accuracy_vs_perturbation(df):

    models = df["Model"].unique()


    for model in models:

        plt.figure(
            figsize=(8,5)
        )


        model_data = df[
            df["Model"] == model
        ]


        for region in model_data["Region"].unique():

            region_data = model_data[
                model_data["Region"] == region
            ]


            plt.plot(
                region_data["Perturbation Percentage"],
                region_data["Accuracy"],
                marker="o",
                label=region
            )


        plt.title(
            f"{model} Robustness Analysis"
        )

        plt.xlabel(
            "Weight Perturbation (%)"
        )

        plt.ylabel(
            "Top-1 Accuracy (%)"
        )


        plt.xticks(
            [5,10,15]
        )


        plt.legend()

        plt.grid()


        output_file = (
            OUTPUT_DIR /
            f"{model.replace('-', '')}_robustness.png"
        )


        plt.savefig(
            output_file,
            bbox_inches="tight"
        )


        plt.close()


        print(
            "Graph saved:"
        )

        print(
            output_file
        )



def plot_model_comparison(df):

    plt.figure(
        figsize=(8,5)
    )


    for model in df["Model"].unique():

        model_data = df[
            df["Model"] == model
        ]


        average_accuracy = (
            model_data
            .groupby(
                "Perturbation Percentage"
            )["Accuracy"]
            .mean()
        )


        plt.plot(
            average_accuracy.index,
            average_accuracy.values,
            marker="o",
            label=model
        )


    plt.title(
        "Overall Model Robustness Comparison"
    )


    plt.xlabel(
        "Weight Perturbation (%)"
    )


    plt.ylabel(
        "Average Accuracy (%)"
    )


    plt.xticks(
        [5,10,15]
    )


    plt.legend()

    plt.grid()


    output_file = (
        OUTPUT_DIR /
        "model_comparison.png"
    )


    plt.savefig(
        output_file,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Comparison graph saved:"
    )

    print(
        output_file
    )



def main():

    create_output_folder()


    df = pd.read_csv(
        INPUT_FILE
    )


    print("\nLoaded Results:")
    print(df)


    generate_summary_table(
        df
    )


    plot_accuracy_vs_perturbation(
        df
    )


    plot_model_comparison(
        df
    )



if __name__ == "__main__":

    main()