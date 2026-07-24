from torchvision import models


def inspect_layers(model, model_name):

    print("=" * 60)
    print(model_name)
    print("=" * 60)

    for name, parameter in model.named_parameters():

        if "weight" in name:

            print(name)


def main():

    resnet50 = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    inceptionv3 = models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT
    )


    inspect_layers(
        resnet50,
        "ResNet-50"
    )


    inspect_layers(
        inceptionv3,
        "InceptionV3"
    )


if __name__ == "__main__":
    main()
