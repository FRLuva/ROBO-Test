"""
Model evaluation pipeline.

Supports:
- ResNet-50
- InceptionV3
"""

import torch

from src.evaluation.dataset_loader import create_dataloader
from src.evaluation.inference import load_model


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def run_inference(model_name):

    dataloader = create_dataloader()

    model = load_model(
        model_name
    )

    predictions = []
    true_labels = []

    model.eval()


    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(
                images
            )


            if isinstance(outputs, tuple):
                outputs = outputs[0]


            predicted_classes = torch.argmax(
                outputs,
                dim=1
            )


            predictions.extend(
                predicted_classes.cpu().tolist()
            )

            true_labels.extend(
                labels.cpu().tolist()
            )


    return predictions, true_labels



def calculate_top1_accuracy(
        predictions,
        true_labels
):

    correct = 0

    total = len(true_labels)


    for prediction, label in zip(
        predictions,
        true_labels
    ):

        if prediction == label:
            correct += 1


    accuracy = (
        correct / total
    ) * 100


    return accuracy



def main():

    predictions, labels = run_inference(
        "inceptionv3"
    )


    print(
        "Total predictions:",
        len(predictions)
    )


    print(
        "First 10 predictions:"
    )

    print(
        predictions[:10]
    )


if __name__ == "__main__":

    main()