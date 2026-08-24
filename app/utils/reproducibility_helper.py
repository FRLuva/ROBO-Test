import hashlib
from pathlib import Path

import torch
from torchvision import models

from src.evaluation.inference import predict_image
from src.perturbation.layer_weight_perturbation import (
    apply_layer_weight_perturbation
)


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
# Model Loading
# ---------------------------------------

def load_selected_model(model_name):

    if model_name == "ResNet-50":

        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

    elif model_name == "InceptionV3":

        model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT
        )

    else:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    model.eval()

    return model


# ---------------------------------------
# Model Weight Hash
# ---------------------------------------

def calculate_model_hash(model):

    hasher = hashlib.sha256()

    for parameter in model.parameters():

        tensor = (
            parameter
            .detach()
            .cpu()
            .contiguous()
        )

        hasher.update(
            tensor.numpy().tobytes()
        )

    return hasher.hexdigest()


# ---------------------------------------
# Run One Reproducibility Experiment
# ---------------------------------------

def run_single_experiment(
    model_name,
    layer_region,
    perturbation_percentage,
    seed,
    image_path
):

    model = load_selected_model(
        model_name
    )

    # -----------------------------
    # Before perturbation
    # -----------------------------

    before_result = predict_image(
        model,
        image_path
    )

    # -----------------------------
    # Apply perturbation
    # -----------------------------

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

    # -----------------------------
    # After perturbation
    # -----------------------------

    after_result = predict_image(
        perturbed_model,
        image_path
    )

    # -----------------------------
    # Hash perturbed model
    # -----------------------------

    model_hash = calculate_model_hash(
        perturbed_model
    )

    return {

        "before": before_result,

        "after": after_result,

        "modified_count": modified_count,

        "model_hash": model_hash

    }


# ---------------------------------------
# Compare Two Runs
# ---------------------------------------

def compare_runs(run1, run2):

    before_prediction_match = (
        run1["before"]["class_index"]
        ==
        run2["before"]["class_index"]
    )

    before_confidence_match = (
        run1["before"]["confidence"]
        ==
        run2["before"]["confidence"]
    )

    after_prediction_match = (
        run1["after"]["class_index"]
        ==
        run2["after"]["class_index"]
    )

    after_confidence_match = (
        run1["after"]["confidence"]
        ==
        run2["after"]["confidence"]
    )

    modified_count_match = (
        run1["modified_count"]
        ==
        run2["modified_count"]
    )

    model_hash_match = (
        run1["model_hash"]
        ==
        run2["model_hash"]
    )

    reproducible = all([
        before_prediction_match,
        before_confidence_match,
        after_prediction_match,
        after_confidence_match,
        modified_count_match,
        model_hash_match
    ])

    return {

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
            model_hash_match,

        "reproducible":
            reproducible

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

    output_folder = Path(
        "results/reproducibility"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    image_name = Path(
        image_path
    ).stem

    safe_model = (
        model_name
        .replace("-", "")
        .replace(" ", "")
    )

    filename = (
        f"{safe_model}_"
        f"{layer_region}_"
        f"{perturbation_percentage}pct_"
        f"seed{seed}_"
        f"{image_name}.txt"
    )

    output_file = (
        output_folder / filename
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "ROBO-Test Interactive "
            "Reproducibility Verification\n"
        )

        file.write(
            "========================================\n\n"
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
            f"Random Seed: {seed}\n"
        )

        file.write(
            f"Evaluation Image: "
            f"{Path(image_path).name}\n\n"
        )

        # --------------------------------
        # Run 1
        # --------------------------------

        file.write(
            "EXPERIMENT 1\n"
        )

        file.write(
            "----------------------------------------\n"
        )

        file.write(
            f"Before Prediction: "
            f"{run1['before']['class_name']}\n"
        )

        file.write(
            f"Before Class Index: "
            f"{run1['before']['class_index']}\n"
        )

        file.write(
            f"Before Confidence: "
            f"{run1['before']['confidence'] * 100:.6f}%\n"
        )

        file.write(
            f"After Prediction: "
            f"{run1['after']['class_name']}\n"
        )

        file.write(
            f"After Class Index: "
            f"{run1['after']['class_index']}\n"
        )

        file.write(
            f"After Confidence: "
            f"{run1['after']['confidence'] * 100:.6f}%\n"
        )

        file.write(
            f"Modified Weights: "
            f"{run1['modified_count']}\n"
        )

        file.write(
            f"Perturbed Model SHA-256: "
            f"{run1['model_hash']}\n\n"
        )

        # --------------------------------
        # Run 2
        # --------------------------------

        file.write(
            "EXPERIMENT 2\n"
        )

        file.write(
            "----------------------------------------\n"
        )

        file.write(
            f"Before Prediction: "
            f"{run2['before']['class_name']}\n"
        )

        file.write(
            f"Before Class Index: "
            f"{run2['before']['class_index']}\n"
        )

        file.write(
            f"Before Confidence: "
            f"{run2['before']['confidence'] * 100:.6f}%\n"
        )

        file.write(
            f"After Prediction: "
            f"{run2['after']['class_name']}\n"
        )

        file.write(
            f"After Class Index: "
            f"{run2['after']['class_index']}\n"
        )

        file.write(
            f"After Confidence: "
            f"{run2['after']['confidence'] * 100:.6f}%\n"
        )

        file.write(
            f"Modified Weights: "
            f"{run2['modified_count']}\n"
        )

        file.write(
            f"Perturbed Model SHA-256: "
            f"{run2['model_hash']}\n\n"
        )

        # --------------------------------
        # Comparison
        # --------------------------------

        file.write(
            "REPRODUCIBILITY COMPARISON\n"
        )

        file.write(
            "========================================\n"
        )

        file.write(
            f"Before Prediction Match: "
            f"{comparison['before_prediction_match']}\n"
        )

        file.write(
            f"Before Confidence Match: "
            f"{comparison['before_confidence_match']}\n"
        )

        file.write(
            f"After Prediction Match: "
            f"{comparison['after_prediction_match']}\n"
        )

        file.write(
            f"After Confidence Match: "
            f"{comparison['after_confidence_match']}\n"
        )

        file.write(
            f"Modified Weight Count Match: "
            f"{comparison['modified_count_match']}\n"
        )

        file.write(
            f"Perturbed Model Hash Match: "
            f"{comparison['model_hash_match']}\n\n"
        )

        if comparison["reproducible"]:

            file.write(
                "FINAL RESULT: PASS\n"
            )

            file.write(
                "The selected experiment produced "
                "identical results across both runs "
                "under the same configuration and "
                "random seed.\n"
            )

        else:

            file.write(
                "FINAL RESULT: FAIL\n"
            )

            file.write(
                "The selected experiment produced "
                "different results across the two runs.\n"
            )

    return output_file