from pathlib import Path

from src.evaluation.inference import load_model, predict_image
from src.perturbation.weight_perturbation import apply_weight_perturbation

image = list(Path("data/evaluation_dataset/golden_retriever").glob("*"))[0]

model = load_model("resnet50")

before = predict_image(model, image)

perturbed_model, _ = apply_weight_perturbation(
    model,
    5,
    seed=42
)

after = predict_image(perturbed_model, image)

print("Before:", before)
print("After :", after)