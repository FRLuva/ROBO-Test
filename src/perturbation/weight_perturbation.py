import torch
import copy


def apply_weight_perturbation(
        model,
        perturbation_percentage,
        seed=None
):
    """
    Randomly sets a percentage of model weights to zero.

    Args:
        model: PyTorch model
        perturbation_percentage: Percentage of weights to remove
        seed: Random seed for reproducibility (optional)

    Returns:
        Perturbed model
        Number of modified weights
    """

    if seed is not None:
        torch.manual_seed(seed)

    perturbed_model = copy.deepcopy(model)

    all_weights = []

    for name, parameter in perturbed_model.named_parameters():

        if "weight" in name:
            all_weights.append(parameter)

    total_weights = sum(
        parameter.numel()
        for parameter in all_weights
    )

    number_of_weights_to_modify = int(
        total_weights * (perturbation_percentage / 100)
    )

    modified_count = 0

    selected_indices = torch.randperm(
        total_weights
    )[:number_of_weights_to_modify]

    current_index = 0

    for parameter in all_weights:

        parameter_size = parameter.numel()

        parameter_indices = selected_indices[
            (selected_indices >= current_index) &
            (selected_indices < current_index + parameter_size)
        ]

        if len(parameter_indices) > 0:

            local_indices = parameter_indices - current_index

            with torch.no_grad():

                flat_parameter = parameter.view(-1)

                flat_parameter[local_indices] = 0

            modified_count += len(local_indices)

        current_index += parameter_size

    return perturbed_model, modified_count