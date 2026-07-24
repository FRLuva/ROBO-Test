from torchvision import models

from src.perturbation.layer_weight_perturbation import (
    apply_layer_weight_perturbation
)


model = models.inception_v3(
    weights=models.Inception_V3_Weights.DEFAULT
)


perturbed_model, modified = apply_layer_weight_perturbation(
    model,
    "Mixed_6",
    5
)


print("\nModified weights:", modified)