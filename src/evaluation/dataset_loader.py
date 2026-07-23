"""
Dataset loader for ImageNet evaluation subset.

Uses original ImageNet class IDs as labels.
"""

from pathlib import Path
import csv

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DATASET_PATH = Path(
    "data/evaluation_dataset"
)

METADATA_PATH = Path(
    "docs/metadata/selected_classes.csv"
)


class ImageNetSubsetDataset(Dataset):

    def __init__(self):

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.images = []
        self.labels = []

        self.load_images()


    def load_images(self):

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                class_name = row["ImageNet Class"]

                folder_id = int(
                    row["Folder ID"]
                )

                folder_name = (
                    class_name
                    .lower()
                    .replace(" ", "_")
                )

                class_folder = (
                    DATASET_PATH / folder_name
                )

                if not class_folder.exists():
                    print(
                        f"Missing folder: {class_folder}"
                    )
                    continue


                for image_path in class_folder.iterdir():

                    if image_path.suffix.lower() in [
                        ".jpg",
                        ".jpeg",
                        ".png"
                    ]:

                        self.images.append(
                            image_path
                        )

                        self.labels.append(
                            folder_id
                        )


    def __len__(self):

        return len(self.images)


    def __getitem__(self, index):

        image_path = self.images[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(
            image
        )

        label = self.labels[index]

        return image, label



def create_dataloader(
        batch_size=32
):

    dataset = ImageNetSubsetDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return dataloader



def main():

    loader = create_dataloader()

    print(
        "Number of images:",
        len(loader.dataset)
    )

    print(
        "First labels:"
    )

    for images, labels in loader:

        print(
            labels[:10]
        )

        break



if __name__ == "__main__":
    main()