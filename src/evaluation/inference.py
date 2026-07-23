"""
Inference pipeline for ImageNet pre-trained models.

This script:
1. Loads a pre-trained model.
2. Loads and preprocesses an image.
3. Performs inference.
4. Returns the predicted ImageNet class index.
"""

from pathlib import Path

import torch
from PIL import Image

import torchvision.models as models
from torchvision import transforms
from torchvision.models import (
    Inception_V3_Weights,
    ResNet50_Weights,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_preprocessing_transform():
    """
    ImageNet preprocessing pipeline.
    """

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_model(model_name):
    """
    Load a pre-trained model.
    """

    if model_name.lower() == "resnet50":

        model = models.resnet50(
            weights=ResNet50_Weights.DEFAULT
        )

    elif model_name.lower() == "inceptionv3":

        model = models.inception_v3(
            weights=Inception_V3_Weights.DEFAULT
        )

    else:
        raise ValueError("Unsupported model.")

    model.eval()
    model.to(DEVICE)

    return model


def predict_image(model, image_path):
    """
    Predict the ImageNet class index of one image.
    """

    transform = get_preprocessing_transform()

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        output = model(image)

        if isinstance(output, tuple):
            output = output[0]

        predicted_class = torch.argmax(
            output,
            dim=1
        ).item()

    return predicted_class


def main():

    image_folder = Path(
        "data/evaluation_dataset/golden_retriever"
    )

    images = list(image_folder.glob("*"))

    if len(images) == 0:
        print("No images found.")
        return

    model = load_model("resnet50")

    prediction = predict_image(
        model,
        images[0]
    )

    print(
        f"Predicted ImageNet Class Index: {prediction}"
    )


if __name__ == "__main__":
    main()