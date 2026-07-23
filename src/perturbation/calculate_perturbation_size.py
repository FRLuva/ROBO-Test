from torchvision import models


def count_model_weights(model):
    total_weights = 0

    for name, parameter in model.named_parameters():
        if "weight" in name:
            total_weights += parameter.numel()

    return total_weights


def calculate_affected_weights(total_weights, perturbation_percentage):
    affected_weights = int(
        total_weights * (perturbation_percentage / 100)
    )

    return affected_weights


def analyze_model(model, model_name):

    total_weights = count_model_weights(model)

    print("=" * 60)
    print(f"Model: {model_name}")
    print("=" * 60)

    print(f"Total weights: {total_weights}")

    for percentage in [5, 10, 15]:

        affected = calculate_affected_weights(
            total_weights,
            percentage
        )

        print(
            f"{percentage}% perturbation affects: {affected} weights"
        )

    print()


def main():

    resnet50 = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    inceptionv3 = models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT
    )


    analyze_model(
        resnet50,
        "ResNet-50"
    )

    analyze_model(
        inceptionv3,
        "InceptionV3"
    )


if __name__ == "__main__":
    main()