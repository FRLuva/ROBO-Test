"""
Save sample predictions.

Supports:
- ResNet-50
- InceptionV3
"""

from pathlib import Path

import torch

from src.evaluation.dataset_loader import (
    ImageNetSubsetDataset
)

from src.evaluation.inference import (
    load_model
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def save_sample_predictions(
        model_name,
        number_of_samples=10
):

    dataset = ImageNetSubsetDataset()

    model = load_model(
        model_name
    )

    model.eval()


    output_file = Path(
        f"results/sample_predictions_{model_name}.txt"
    )


    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:


        file.write(
            f"{model_name} Sample Predictions\n"
        )

        file.write(
            "=" * 50 + "\n\n"
        )


        with torch.no_grad():

            for index in range(
                min(
                    number_of_samples,
                    len(dataset)
                )
            ):

                image, true_label = dataset[index]


                image = image.unsqueeze(
                    0
                )

                image = image.to(
                    DEVICE
                )


                output = model(
                    image
                )


                if isinstance(output, tuple):

                    output = output[0]


                predicted_label = torch.argmax(
                    output,
                    dim=1
                ).item()


                file.write(
                    f"Image Index: {index}\n"
                )

                file.write(
                    f"Image Path: {dataset.images[index]}\n"
                )

                file.write(
                    f"True Label: {true_label}\n"
                )

                file.write(
                    f"Predicted Label: {predicted_label}\n"
                )

                file.write(
                    "-" * 50 + "\n"
                )


    print(
        "Sample predictions saved:"
    )

    print(
        output_file
    )



def main():

    save_sample_predictions(
        "inceptionv3"
    )


if __name__ == "__main__":

    main()