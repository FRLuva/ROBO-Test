"""
Calculate baseline Top-1 accuracy for InceptionV3.
"""

from src.evaluation.evaluate_model import (
    run_inference,
    calculate_top1_accuracy
)


def main():

    predictions, labels = run_inference(
        "inceptionv3"
    )


    accuracy = calculate_top1_accuracy(
        predictions,
        labels
    )


    print(
        "InceptionV3 Baseline Top-1 Accuracy:"
    )

    print(
        f"{accuracy:.2f}%"
    )


if __name__ == "__main__":

    main()