import torch
from torchvision import models

from weight_perturbation import apply_weight_perturbation


def compare_models(original_model, perturbed_model):

    changed_weights = 0
    zeroed_weights = 0
    unchanged_weights = 0

    for (original_name, original_param), (perturbed_name, perturbed_param) in zip(
        original_model.named_parameters(),
        perturbed_model.named_parameters()
    ):

        if original_name != perturbed_name:
            print("Parameter mismatch detected!")
            return


        if "weight" in original_name:

            changed = original_param != perturbed_param
            unchanged = original_param == perturbed_param
            zeroed = perturbed_param == 0


            changed_weights += changed.sum().item()
            unchanged_weights += unchanged.sum().item()
            zeroed_weights += zeroed.sum().item()


    return changed_weights, zeroed_weights, unchanged_weights



def verify_architecture(original_model, perturbed_model):

    original_parameters = list(
        original_model.state_dict().keys()
    )

    perturbed_parameters = list(
        perturbed_model.state_dict().keys()
    )


    return original_parameters == perturbed_parameters



def main():

    print("Loading ResNet-50...")

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )


    print("Applying 10% perturbation...")

    perturbed_model, modified_count = apply_weight_perturbation(
        model,
        10
    )


    changed, zeroed, unchanged = compare_models(
        model,
        perturbed_model
    )


    architecture_same = verify_architecture(
        model,
        perturbed_model
    )


    print("\n" + "=" * 60)
    print("Final Perturbation Validation")
    print("=" * 60)


    print(f"Expected modified weights: {modified_count}")
    print(f"Changed weights: {changed}")
    print(f"Zeroed weights: {zeroed}")
    print(f"Unchanged weights: {unchanged}")

    print(
        f"Architecture unchanged: {architecture_same}"
    )


    if (
        changed == modified_count
        and zeroed == modified_count
        and architecture_same
    ):

        print("Validation Status: PASS")

    else:

        print("Validation Status: CHECK")



if __name__ == "__main__":
    main()