import torch
from torchvision import models


def inspect_model_parameters(model, model_name):
    print("=" * 60)
    print(f"Model Parameter Inspection: {model_name}")
    print("=" * 60)

    total_parameters = 0
    trainable_parameters = 0

    print("\nParameter Details:\n")

    for name, param in model.named_parameters():
        parameter_count = param.numel()
        total_parameters += parameter_count

        if param.requires_grad:
            trainable_parameters += parameter_count

        print(f"Name: {name}")
        print(f"Shape: {tuple(param.shape)}")
        print(f"Number of weights: {parameter_count}")
        print("-" * 40)

    print("\nSummary:")
    print(f"Total parameters: {total_parameters}")
    print(f"Trainable parameters: {trainable_parameters}")
    print("=" * 60)


def main():

    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    inceptionv3 = models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT
    )

    inspect_model_parameters(resnet50, "ResNet-50")

    inspect_model_parameters(inceptionv3, "InceptionV3")


if __name__ == "__main__":
    main()