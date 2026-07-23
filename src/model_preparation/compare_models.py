from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    inception_v3,
    Inception_V3_Weights,
)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def count_layers(model):
    return len(list(model.modules()))


def compare_models():
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)

    models = {
        "ResNet-50": resnet,
        "InceptionV3": inception,
    }

    for name, model in models.items():
        total, trainable, non_trainable = count_parameters(model)

        print("=" * 50)
        print(f"Model: {name}")
        print("=" * 50)
        print(f"Number of layers       : {count_layers(model)}")
        print(f"Total parameters       : {total:,}")
        print(f"Trainable parameters   : {trainable:,}")
        print(f"Non-trainable parameters: {non_trainable:,}")
        print(f"Input image size       : {'224 x 224' if name == 'ResNet-50' else '299 x 299'}")
        print(f"Output classes         : 1000")
        print()


if __name__ == "__main__":
    compare_models()