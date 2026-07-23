import torch
from torchvision import models

from weight_perturbation import apply_weight_perturbation


def save_original_weights(model):
    """
    Stores a copy of all model weights.
    """

    original_weights = {}

    for name, parameter in model.named_parameters():

        if "weight" in name:
            original_weights[name] = parameter.clone()

    return original_weights


def compare_weights(original_weights, perturbed_model):

    total_changed = 0
    total_zeroed = 0
    total_unchanged = 0

    for name, parameter in perturbed_model.named_parameters():

        if "weight" in name:

            original = original_weights[name]

            changed_positions = (
                original != parameter
            )

            unchanged_positions = (
                original == parameter
            )

            zero_positions = (
                parameter == 0
            )


            total_changed += changed_positions.sum().item()
            total_unchanged += unchanged_positions.sum().item()
            total_zeroed += zero_positions.sum().item()


    print("=" * 60)
    print("Perturbation Verification Results")
    print("=" * 60)

    print(f"Changed weights: {total_changed}")
    print(f"Zeroed weights: {total_zeroed}")
    print(f"Unchanged weights: {total_unchanged}")


def main():

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    print("Saving original weights...")

    original_weights = save_original_weights(model)


    print("Applying perturbation...")

    perturbed_model, modified_count = apply_weight_perturbation(
        model,
        5
    )


    print(
        f"Expected modified weights: {modified_count}"
    )


    compare_weights(
        original_weights,
        perturbed_model
    )


if __name__ == "__main__":
    main()