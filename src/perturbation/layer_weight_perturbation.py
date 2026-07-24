import torch
import copy


def apply_layer_weight_perturbation(
        model,
        layer_keyword,
        perturbation_percentage,
        seed=None
):
    """
    Applies random weight perturbation to selected layers only.

    Args:
        model: PyTorch model
        layer_keyword: keyword to identify target layers
        perturbation_percentage: percentage of selected layer weights to zero
        seed: random seed

    Returns:
        perturbed model
        modified weight count
    """

    if seed is not None:
        torch.manual_seed(seed)


    perturbed_model = copy.deepcopy(model)


    selected_weights = []
    selected_names = []


    for name, parameter in perturbed_model.named_parameters():

        if (
            layer_keyword.lower() in name.lower()
            and "weight" in name.lower()
            and "bn" not in name.lower()
        ):
            selected_weights.append(parameter)
            selected_names.append(name)


    print("\nSelected layers:")

    if len(selected_names) == 0:
        print("No layers matched.")

    else:
        for name in selected_names:
            print(name)


    total_weights = sum(
        parameter.numel()
        for parameter in selected_weights
    )


    number_to_modify = int(
        total_weights * (perturbation_percentage / 100)
    )


    if number_to_modify == 0:

        return perturbed_model, 0


    selected_indices = torch.randperm(
        total_weights
    )[:number_to_modify]


    modified_count = 0

    current_index = 0


    for parameter in selected_weights:

        parameter_size = parameter.numel()


        parameter_indices = selected_indices[
            (selected_indices >= current_index)
            &
            (selected_indices < current_index + parameter_size)
        ]


        if len(parameter_indices) > 0:

            local_indices = (
                parameter_indices - current_index
            )


            with torch.no_grad():

                flat_parameter = parameter.view(-1)

                flat_parameter[local_indices] = 0


            modified_count += len(local_indices)


        current_index += parameter_size


    print(
        f"Total selected weights: {total_weights}"
    )

    print(
        f"Weights modified: {modified_count}"
    )


    return perturbed_model, modified_count