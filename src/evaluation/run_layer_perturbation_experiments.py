import csv
from pathlib import Path

from torchvision import models

from src.perturbation.layer_weight_perturbation import (
    apply_layer_weight_perturbation
)

from src.evaluation.dataset_loader import (
    create_dataloader
)

from src.evaluation.evaluate_model import (
    calculate_top1_accuracy
)

import torch


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


RESULT_FILE = Path(
    "results/layer_perturbation_results.csv"
)


PERTURBATION_LEVELS = [
    5,
    10,
    15
]


MODEL_CONFIGS = {

    "ResNet-50": {

        "loader": lambda: models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        ),

        "layers": {
            "Early": "conv1",
            "Middle": "layer2",
            "Final": "fc"
        }
    },


    "InceptionV3": {

        "loader": lambda: models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT
        ),

        "layers": {
            "Early": "Conv2d",
            "Middle": "Mixed_6",
            "Final": "fc"
        }
    }

}



def evaluate_perturbed_model(model):

    dataloader = create_dataloader()

    model.eval()

    model.to(DEVICE)


    predictions = []
    labels_list = []


    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(
                images
            )


            if isinstance(outputs, tuple):

                outputs = outputs[0]


            predicted = torch.argmax(
                outputs,
                dim=1
            )


            predictions.extend(
                predicted.cpu().tolist()
            )

            labels_list.extend(
                labels.cpu().tolist()
            )


    accuracy = calculate_top1_accuracy(
        predictions,
        labels_list
    )


    return accuracy



def run_experiment():

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    results = []


    for model_name, config in MODEL_CONFIGS.items():


        print(
            f"\nLoading {model_name}..."
        )


        base_model = config["loader"]()



        for region, keyword in config["layers"].items():


            for percentage in PERTURBATION_LEVELS:


                print(
                    f"{model_name} | {region} | {percentage}%"
                )


                perturbed_model, modified_count = (
                    apply_layer_weight_perturbation(
                        base_model,
                        keyword,
                        percentage,
                        seed=42
                    )
                )


                accuracy = evaluate_perturbed_model(
                    perturbed_model
                )


                results.append({

                    "Model": model_name,

                    "Region": region,

                    "Perturbation Percentage": percentage,

                    "Modified Weights": modified_count,

                    "Accuracy": accuracy

                })



    with open(
        RESULT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:


        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Model",
                "Region",
                "Perturbation Percentage",
                "Modified Weights",
                "Accuracy"
            ]
        )


        writer.writeheader()

        writer.writerows(
            results
        )


    print(
        "\nResults saved:"
    )

    print(
        RESULT_FILE
    )



if __name__ == "__main__":

    run_experiment()