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

def load_class_names():
    """
    Load ImageNet class names.
    """

    class_file = Path("docs/metadata/imagenet_classes.txt")

    with open(class_file, "r") as f:
        classes = [line.strip() for line in f]

    return classes

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

        probabilities = torch.nn.functional.softmax(
             output,
             dim=1
        )

        confidence, predicted_class = torch.max(
             probabilities,
            dim=1
        )

        class_names = load_class_names()

        return {
             "class_index": predicted_class.item(),
             "class_name": class_names[predicted_class.item()],
             "confidence": confidence.item()
        }


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

    print(prediction)


if __name__ == "__main__":
    main()