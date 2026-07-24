import torch
from torchvision import models

from src.perturbation.weight_perturbation import apply_weight_perturbation

def count_zero_weights(model):

    zero_weights = 0
    total_weights = 0

    for name, parameter in model.named_parameters():

        if "weight" in name:

            zero_weights += (parameter == 0).sum().item()
            total_weights += parameter.numel()

    return zero_weights, total_weights



def main():

    print("Loading InceptionV3...")

    model = models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT
    )


    for percentage in [5, 10, 15]:

        print("\n" + "=" * 60)
        print(f"Testing {percentage}% Perturbation")
        print("=" * 60)


        perturbed_model, modified_count = apply_weight_perturbation(
            model,
            percentage
        )


        zero_weights, total_weights = count_zero_weights(
            perturbed_model
        )


        expected_weights = int(
            total_weights * (percentage / 100)
        )


        print(f"Expected modified weights: {expected_weights}")
        print(f"Actual modified weights: {modified_count}")
        print(f"Zero weights after perturbation: {zero_weights}")
        print(f"Total weights: {total_weights}")


        if modified_count == expected_weights:

            print("Status: PASS")

        else:

            print("Status: CHECK")



if __name__ == "__main__":
    main()