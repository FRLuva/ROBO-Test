from torchvision.models import inception_v3, Inception_V3_Weights


def load_model():
    # Load the pre-trained InceptionV3 model
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)

    # Set the model to evaluation mode
    model.eval()

    return model


if __name__ == "__main__":
    model = load_model()
    print("InceptionV3 loaded successfully!")