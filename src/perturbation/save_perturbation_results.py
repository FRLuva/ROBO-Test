import os
from torchvision import models

from weight_perturbation import apply_weight_perturbation


def compare_weights(original_model, perturbed_model):

    changed_weights = 0
    zeroed_weights = 0
    unchanged_weights = 0

    for original_param, perturbed_param in zip(
        original_model.parameters(),
        perturbed_model.parameters()
    ):

        changed = original_param != perturbed_param
        unchanged = original_param == perturbed_param
        zeroed = perturbed_param == 0

        changed_weights += changed.sum().item()
        unchanged_weights += unchanged.sum().item()
        zeroed_weights += zeroed.sum().item()


    return changed_weights, zeroed_weights, unchanged_weights



def save_results():

    os.makedirs(
        "results",
        exist_ok=True
    )


    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )


    perturbation_percentage = 10


    perturbed_model, modified_count = apply_weight_perturbation(
        model,
        perturbation_percentage
    )


    changed, zeroed, unchanged = compare_weights(
        model,
        perturbed_model
    )


    output_file = (
        "results/perturbation_verification.txt"
    )


    with open(output_file, "w") as file:

        file.write(
            "Weight Perturbation Verification Report\n"
        )

        file.write(
            "=" * 50 + "\n\n"
        )

        file.write(
            "Model: ResNet-50\n"
        )

        file.write(
            f"Perturbation Percentage: {perturbation_percentage}%\n\n"
        )

        file.write(
            f"Expected Modified Weights: {modified_count}\n"
        )

        file.write(
            f"Changed Weights: {changed}\n"
        )

        file.write(
            f"Zeroed Weights: {zeroed}\n"
        )

        file.write(
            f"Unchanged Weights: {unchanged}\n"
        )

        if changed == modified_count and zeroed == modified_count:

            file.write(
                "\nValidation Status: PASS\n"
            )

        else:

            file.write(
                "\nValidation Status: CHECK\n"
            )


    print(
        f"Results saved to {output_file}"
    )



if __name__ == "__main__":
    save_results()