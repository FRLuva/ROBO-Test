from torchvision import transforms
from pathlib import Path
from PIL import Image

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

TEST_IMAGE = Path(
    "data/evaluation_dataset/golden_retriever/ILSVRC2012_val_00001112.JPEG"
)

image = Image.open(TEST_IMAGE).convert("RGB")

processed_image = preprocess(image)

print(f"Tensor shape: {processed_image.shape}")
print(f"Tensor data type: {processed_image.dtype}")
print(f"Minimum value: {processed_image.min():.4f}")
print(f"Maximum value: {processed_image.max():.4f}")
