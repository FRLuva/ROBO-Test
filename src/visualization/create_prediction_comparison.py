import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from src.evaluation.inference import (
    load_model,
    predict_image
)

from src.perturbation.weight_perturbation import (
    apply_weight_perturbation
)


def create_comparison():

    output_folder = "figures/prediction_comparisons"
    os.makedirs(output_folder, exist_ok=True)

    image_folder = Path(
        "data/evaluation_dataset/golden_retriever"
    )

    images = list(image_folder.glob("*"))

    if len(images) == 0:
        print("No images found.")
        return

    image_path = images[0]

    models = [
        "resnet50",
        "inceptionv3"
    ]

    for model_name in models:

        print(f"Processing {model_name}...")

        # Load original model
        model = load_model(model_name)

        # Prediction before perturbation
        before_prediction = predict_image(
            model,
            image_path
        )

        # Apply 10% perturbation
        perturbed_model, _ = apply_weight_perturbation(
            model,
            10,
            seed=42
        )

        # Prediction after perturbation
        after_prediction = predict_image(
            perturbed_model,
            image_path
        )

        # Load image for display
        image = Image.open(image_path)

        # Create side-by-side figure
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"Before\nPrediction: {before_prediction}"
        )

        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"After 10% Perturbation\nPrediction: {after_prediction}"
        )

        output_file = (
            f"{output_folder}/{model_name}_before_after.png"
        )

        plt.savefig(
            output_file,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {output_file}")


if __name__ == "__main__":
    create_comparison()