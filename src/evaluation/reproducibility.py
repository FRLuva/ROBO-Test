from pathlib import Path

import torch
from torchvision import models

from src.perturbation.layer_weight_perturbation import (
    apply_layer_weight_perturbation
)

from src.evaluation.inference import predict_image


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------
# Model Loaders
# ---------------------------------------

MODEL_LOADERS = {

    "ResNet-50": lambda: models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    ),

    "InceptionV3": lambda: models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT
    )

}


# ---------------------------------------
# Layer Mapping
# ---------------------------------------

LAYER_MAPPING = {

    "ResNet-50": {

        "Early": "conv1",
        "Middle": "layer2",
        "Final": "fc"

    },

    "InceptionV3": {

        "Early": "Conv2d",
        "Middle": "Mixed_6",
        "Final": "fc"

    }

}


# ---------------------------------------
# Calculate Model Checksum
# ---------------------------------------

def calculate_checksum(model):
    """
    Calculate a deterministic checksum
    of all model parameters.
    """

    checksum = 0.0

    with torch.no_grad():

        for parameter in model.parameters():

            checksum += (
                parameter
                .double()
                .sum()
                .item()
            )

    return checksum


# ---------------------------------------
# Run Single Experiment
# ---------------------------------------

def run_single_experiment(
    model_name,
    layer_region,
    perturbation_percentage,
    seed,
    image_path
):
    """
    Run one complete reproducibility experiment.

    The experiment:

    1. Loads the selected pretrained model.
    2. Predicts the image before perturbation.
    3. Applies the selected layer perturbation.
    4. Predicts the image after perturbation.
    5. Calculates the modified weight count.
    6. Calculates a checksum of the perturbed model.

    Returns all relevant results.
    """

    # -----------------------------------
    # Load model
    # -----------------------------------

    model = MODEL_LOADERS[model_name]()

    model.eval()

    model.to(DEVICE)


    # -----------------------------------
    # BEFORE prediction
    # -----------------------------------

    before_result = predict_image(
        model,
        image_path
    )


    # -----------------------------------
    # Apply perturbation
    # -----------------------------------

    layer_keyword = LAYER_MAPPING[
        model_name
    ][layer_region]


    perturbed_model, modified_count = (
        apply_layer_weight_perturbation(
            model,
            layer_keyword,
            perturbation_percentage,
            seed=seed
        )
    )


    perturbed_model.eval()

    perturbed_model.to(DEVICE)


    # -----------------------------------
    # AFTER prediction
    # -----------------------------------

    after_result = predict_image(
        perturbed_model,
        image_path
    )


    # -----------------------------------
    # Model checksum
    # -----------------------------------

    checksum = calculate_checksum(
        perturbed_model
    )


    # -----------------------------------
    # Return results
    # -----------------------------------

    return {

        "model": model_name,

        "layer_region": layer_region,

        "perturbation": perturbation_percentage,

        "seed": seed,

        "image": str(image_path),

        "before": before_result,

        "after": after_result,

        "modified_count": modified_count,

        "checksum": checksum

    }


# ---------------------------------------
# Compare Two Runs
# ---------------------------------------

def compare_runs(
    run1,
    run2
):
    """
    Compare two reproducibility experiments.
    """

    # -----------------------------------
    # Before prediction
    # -----------------------------------

    before_prediction_match = (
        run1["before"]["class_index"]
        ==
        run2["before"]["class_index"]
    )


    # -----------------------------------
    # Before confidence
    # -----------------------------------

    before_confidence_match = (
        abs(
            run1["before"]["confidence"]
            -
            run2["before"]["confidence"]
        )
        < 1e-6
    )


    # -----------------------------------
    # After prediction
    # -----------------------------------

    after_prediction_match = (
        run1["after"]["class_index"]
        ==
        run2["after"]["class_index"]
    )


    # -----------------------------------
    # After confidence
    # -----------------------------------

    after_confidence_match = (
        abs(
            run1["after"]["confidence"]
            -
            run2["after"]["confidence"]
        )
        < 1e-6
    )


    # -----------------------------------
    # Modified weights
    # -----------------------------------

    modified_count_match = (
        run1["modified_count"]
        ==
        run2["modified_count"]
    )


    # -----------------------------------
    # Perturbed model checksum
    # -----------------------------------

    model_hash_match = (
        abs(
            run1["checksum"]
            -
            run2["checksum"]
        )
        < 1e-6
    )


    # -----------------------------------
    # Overall result
    # -----------------------------------

    reproducible = (
        before_prediction_match
        and
        before_confidence_match
        and
        after_prediction_match
        and
        after_confidence_match
        and
        modified_count_match
        and
        model_hash_match
    )


    return {

        "reproducible":
            reproducible,

        "before_prediction_match":
            before_prediction_match,

        "before_confidence_match":
            before_confidence_match,

        "after_prediction_match":
            after_prediction_match,

        "after_confidence_match":
            after_confidence_match,

        "modified_count_match":
            modified_count_match,

        "model_hash_match":
            model_hash_match

    }


# ---------------------------------------
# Save Verification Result
# ---------------------------------------

def save_verification_result(
    model_name,
    layer_region,
    perturbation_percentage,
    seed,
    image_path,
    run1,
    run2,
    comparison
):
    """
    Save the reproducibility verification
    result to a TXT file.
    """

    output_file = Path(
        "results/reproducibility_verification.txt"
    )


    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    status = (
        "PASS"
        if comparison["reproducible"]
        else
        "FAIL"
    )


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:


        file.write(
            "======================================\n"
        )

        file.write(
            "ROBO-Test Reproducibility Verification\n"
        )

        file.write(
            "======================================\n\n"
        )


        # -----------------------------------
        # Configuration
        # -----------------------------------

        file.write(
            "Experiment Configuration\n"
        )

        file.write(
            "--------------------------------------\n"
        )

        file.write(
            f"Model: {model_name}\n"
        )

        file.write(
            f"Layer Region: {layer_region}\n"
        )

        file.write(
            f"Perturbation: "
            f"{perturbation_percentage}%\n"
        )

        file.write(
            f"Seed: {seed}\n"
        )

        file.write(
            f"Image: {image_path}\n\n"
        )


        # -----------------------------------
        # Experiment 1
        # -----------------------------------

        file.write(
            "Experiment 1\n"
        )

        file.write(
            "--------------------------------------\n"
        )

        file.write(
            f"Before Prediction: "
            f"{run1['before']['class_name']}\n"
        )

        file.write(
            f"Before Confidence: "
            f"{run1['before']['confidence']:.6f}\n"
        )

        file.write(
            f"After Prediction: "
            f"{run1['after']['class_name']}\n"
        )

        file.write(
            f"After Confidence: "
            f"{run1['after']['confidence']:.6f}\n"
        )

        file.write(
            f"Modified Weights: "
            f"{run1['modified_count']}\n"
        )

        file.write(
            f"Checksum: "
            f"{run1['checksum']}\n\n"
        )


        # -----------------------------------
        # Experiment 2
        # -----------------------------------

        file.write(
            "Experiment 2\n"
        )

        file.write(
            "--------------------------------------\n"
        )

        file.write(
            f"Before Prediction: "
            f"{run2['before']['class_name']}\n"
        )

        file.write(
            f"Before Confidence: "
            f"{run2['before']['confidence']:.6f}\n"
        )

        file.write(
            f"After Prediction: "
            f"{run2['after']['class_name']}\n"
        )

        file.write(
            f"After Confidence: "
            f"{run2['after']['confidence']:.6f}\n"
        )

        file.write(
            f"Modified Weights: "
            f"{run2['modified_count']}\n"
        )

        file.write(
            f"Checksum: "
            f"{run2['checksum']}\n\n"
        )


        # -----------------------------------
        # Verification
        # -----------------------------------

        file.write(
            "Verification Checks\n"
        )

        file.write(
            "--------------------------------------\n"
        )

        file.write(
            f"Before Prediction Match: "
            f"{'PASS' if comparison['before_prediction_match'] else 'FAIL'}\n"
        )

        file.write(
            f"Before Confidence Match: "
            f"{'PASS' if comparison['before_confidence_match'] else 'FAIL'}\n"
        )

        file.write(
            f"After Prediction Match: "
            f"{'PASS' if comparison['after_prediction_match'] else 'FAIL'}\n"
        )

        file.write(
            f"After Confidence Match: "
            f"{'PASS' if comparison['after_confidence_match'] else 'FAIL'}\n"
        )

        file.write(
            f"Modified Weight Count Match: "
            f"{'PASS' if comparison['modified_count_match'] else 'FAIL'}\n"
        )

        file.write(
            f"Perturbed Model Checksum Match: "
            f"{'PASS' if comparison['model_hash_match'] else 'FAIL'}\n\n"
        )


        # -----------------------------------
        # Final Result
        # -----------------------------------

        file.write(
            "======================================\n"
        )

        file.write(
            f"RESULT: {status}\n"
        )

        file.write(
            "======================================\n"
        )


    return output_file