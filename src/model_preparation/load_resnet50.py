from torchvision.models import resnet50, ResNet50_Weights


def load_model():
    # Load the pre-trained ResNet-50 model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Set the model to evaluation mode
    model.eval()

    return model


if __name__ == "__main__":
    model = load_model()
    print("ResNet-50 loaded successfully!")