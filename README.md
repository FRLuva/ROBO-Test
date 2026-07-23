# ROBO-Test
ROBO Test evaluates the robustness of pre-trained neural networks (ResNet-50 and InceptionV3) against random weight perturbations. The project measures accuracy loss at different perturbation levels and across different layer positions to identify which architecture better withstands weight corruption and maintains performance.

## Evaluation Dataset

The evaluation dataset consists of a publicly available ImageNet validation subset. 
Twenty ImageNet classes were selected for robustness evaluation, with 5 images per class, resulting in a total of 100 evaluation images.

The selected classes represent different categories including vehicles, household objects, food items, sports objects, and animals. 
The dataset is used consistently across all experiments to ensure a fair comparison between different pretrained neural network architectures.