"""
Save baseline evaluation results.

Supports:
- ResNet-50
- InceptionV3
"""

from pathlib import Path

from src.evaluation.evaluate_model import (
    run_inference,
    calculate_top1_accuracy
)


def save_results(model_name):

    predictions, labels = run_inference(
        model_name
    )

    accuracy = calculate_top1_accuracy(
        predictions,
        labels
    )


    result_file = Path(
        f"results/baseline_{model_name}.txt"
    )


    result_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        result_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Model: {model_name}\n"
        )

        file.write(
            f"Top-1 Accuracy: {accuracy:.2f}%\n"
        )

        file.write(
            f"Total Images: {len(labels)}\n"
        )


    print(
        "Baseline results saved:"
    )

    print(
        result_file
    )


def main():

    save_results(
        "inceptionv3"
    )


if __name__ == "__main__":

    main()