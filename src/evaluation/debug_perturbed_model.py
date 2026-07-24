from torchvision import models

from src.perturbation.weight_perturbation import apply_weight_perturbation
from src.evaluation.evaluate_model import (
    run_inference,
    calculate_top1_accuracy
)


def evaluate(model, title):

    predictions, labels = run_inference(
        model=model
    )

    accuracy = calculate_top1_accuracy(
        predictions,
        labels
    )

    print(f"\n{title}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("First 20 predictions:")
    print(predictions[:20])
    print("First 20 labels:")
    print(labels[:20])


def main():

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    evaluate(
        model,
        "Original ResNet-50"
    )

    perturbed_model, modified = apply_weight_perturbation(
        model,
        5,
        seed=42
    )

    print(f"\nModified weights: {modified}")

    evaluate(
        perturbed_model,
        "Perturbed ResNet-50"
    )


if __name__ == "__main__":
    main()