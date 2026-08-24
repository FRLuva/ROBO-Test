import sys
from pathlib import Path

# ---------------------------------------
# Add project root to Python path
# ---------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


import torch
from torchvision import models

from src.perturbation.layer_weight_perturbation import (
    apply_layer_weight_perturbation
)


# ---------------------------------------
# Configuration
# ---------------------------------------

MODEL_NAME = "ResNet-50"
LAYER_REGION = "Middle"
LAYER_KEYWORD = "layer2"

PERTURBATION_PERCENTAGE = 10
SEED = 42

RESULT_FILE = Path(
    "results/reproducibility_verification.txt"
)


# ---------------------------------------
# Load Model
# ---------------------------------------

def load_resnet50():

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    model.eval()

    return model


# ---------------------------------------
# Calculate Model Weight Checksum
# ---------------------------------------

def calculate_weight_checksum(model):

    checksum = 0.0

    for parameter in model.parameters():

        checksum += parameter.detach().float().sum().item()

    return checksum


# ---------------------------------------
# Run One Experiment
# ---------------------------------------

def run_experiment():

    model = load_resnet50()

    perturbed_model, modified_count = (
        apply_layer_weight_perturbation(
            model,
            LAYER_KEYWORD,
            PERTURBATION_PERCENTAGE,
            seed=SEED
        )
    )

    checksum = calculate_weight_checksum(
        perturbed_model
    )

    return modified_count, checksum


# ---------------------------------------
# Main Reproducibility Test
# ---------------------------------------

def main():

    print("\n======================================")
    print("ROBO-Test Reproducibility Verification")
    print("======================================")

    print(f"Model: {MODEL_NAME}")
    print(f"Layer Region: {LAYER_REGION}")
    print(f"Perturbation: {PERTURBATION_PERCENTAGE}%")
    print(f"Seed: {SEED}")

    print("\nRunning Experiment 1...")

    modified_1, checksum_1 = run_experiment()

    print(f"Modified weights: {modified_1}")
    print(f"Checksum: {checksum_1}")

    print("\nRunning Experiment 2...")

    modified_2, checksum_2 = run_experiment()

    print(f"Modified weights: {modified_2}")
    print(f"Checksum: {checksum_2}")

    # ---------------------------------------
    # Compare Results
    # ---------------------------------------

    modified_match = (
        modified_1 == modified_2
    )

    checksum_match = (
        checksum_1 == checksum_2
    )

    reproducible = (
        modified_match
        and checksum_match
    )

    # ---------------------------------------
    # Display Result
    # ---------------------------------------

    print("\n======================================")

    if reproducible:

        print("RESULT: PASS")
        print("Experiment is reproducible.")

    else:

        print("RESULT: FAIL")
        print("Experiment is NOT reproducible.")

    print("======================================")

    # ---------------------------------------
    # Save Verification
    # ---------------------------------------

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "ROBO-Test Reproducibility Verification\n"
        )

        file.write(
            "======================================\n\n"
        )

        file.write(
            f"Model: {MODEL_NAME}\n"
        )

        file.write(
            f"Layer Region: {LAYER_REGION}\n"
        )

        file.write(
            f"Layer Keyword: {LAYER_KEYWORD}\n"
        )

        file.write(
            f"Perturbation: {PERTURBATION_PERCENTAGE}%\n"
        )

        file.write(
            f"Random Seed: {SEED}\n\n"
        )

        file.write(
            "Experiment 1\n"
        )

        file.write(
            f"Modified Weights: {modified_1}\n"
        )

        file.write(
            f"Weight Checksum: {checksum_1}\n\n"
        )

        file.write(
            "Experiment 2\n"
        )

        file.write(
            f"Modified Weights: {modified_2}\n"
        )

        file.write(
            f"Weight Checksum: {checksum_2}\n\n"
        )

        file.write(
            f"Modified Weight Match: "
            f"{modified_match}\n"
        )

        file.write(
            f"Checksum Match: "
            f"{checksum_match}\n\n"
        )

        if reproducible:

            file.write(
                "FINAL RESULT: PASS\n"
            )

            file.write(
                "The experiment produced identical "
                "results under the same configuration "
                "and random seed.\n"
            )

        else:

            file.write(
                "FINAL RESULT: FAIL\n"
            )

            file.write(
                "The experiment produced different "
                "results under the same configuration.\n"
            )

    print(
        f"\nVerification saved to: {RESULT_FILE}"
    )


if __name__ == "__main__":

    main()