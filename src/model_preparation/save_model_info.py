from pathlib import Path
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


def get_model_info(name, model):
    total, trainable, non_trainable = count_parameters(model)

    input_size = "224 x 224" if name == "ResNet-50" else "299 x 299"

    return f"""
Model: {name}
--------------------------------------------------
Number of layers        : {count_layers(model)}
Total parameters        : {total:,}
Trainable parameters    : {trainable:,}
Non-trainable parameters: {non_trainable:,}
Input image size        : {input_size}
Output classes          : 1000
"""


def main():
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)

    report = "MODEL COMPARISON\n"
    report += "=" * 50 + "\n"
    report += get_model_info("ResNet-50", resnet)
    report += get_model_info("InceptionV3", inception)

    docs_folder = Path("docs")
    docs_folder.mkdir(exist_ok=True)

    output_file = docs_folder / "model_info.txt"

    with open(output_file, "w") as file:
        file.write(report)

    print(f"Model information saved to: {output_file}")


if __name__ == "__main__":
    main()