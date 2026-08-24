# ROBO-Test
ROBO Test evaluates the robustness of pre-trained neural networks (ResNet-50 and InceptionV3) against random weight perturbations. The project measures accuracy loss at different perturbation levels and across different layer positions to identify which architecture better withstands weight corruption and maintains performance.

## Evaluation Dataset

The evaluation dataset consists of a publicly available ImageNet validation subset. 
Twenty ImageNet classes were selected for robustness evaluation, with 5 images per class, resulting in a total of 100 evaluation images.

The selected classes represent different categories including vehicles, household objects, food items, sports objects, and animals. 
The dataset is used consistently across all experiments to ensure a fair comparison between different pretrained neural network architectures.

# ROBO-Test — User Guide

## 1. Project Overview

ROBO-Test is a robustness analysis project for studying the behavior of pre-trained convolutional neural networks when their learned weights are randomly perturbed.

The project evaluates two ImageNet-pretrained models:

* ResNet-50
* InceptionV3

The models are evaluated on a controlled ImageNet-1K subset containing 20 classes and 100 images in total. After establishing baseline performance, the project randomly zeroes out 5%, 10%, and 15% of model weights and measures the resulting changes in classification performance.

The project also provides layer sensitivity analysis, robustness comparison, prediction visualizations, numerical result files, reproducibility testing, and an interactive Streamlit dashboard.

---

## 2. Project Structure

The main project directories are organized as follows:

```text
ROBO-Test/
│
├── app/
│   ├── app.py
│   └── pages/
│       └── Analysis_Dashboard.py
│
├── data/
│   └── evaluation_dataset/
│
├── figures/
│   ├── accuracy_vs_perturbation.png
│   ├── layer_sensitivity.png
│   ├── robustness_comparison.png
│   └── prediction_comparisons/
│       ├── resnet50_before_after.png
│       └── inceptionv3_before_after.png
│
├── results/
│   └── analysis/
│       └── CSV result files
│
├── src/
│   ├── dataset/
│   ├── preprocessing/
│   ├── model_preparation/
│   ├── perturbation/
│   ├── evaluation/
│   ├── analysis/
│   └── visualization/
│
└── README.md
```

---

## 3. Requirements

The project uses Python and PyTorch-based machine learning libraries.

A virtual environment is recommended.

### Create a virtual environment

On Windows:

```bash
python -m venv venv
```

### Activate the virtual environment

PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
venv\Scripts\activate
```

After activation, the terminal should indicate that the virtual environment is active.

---

## 4. Install Dependencies

From the project root, install the required Python packages using the project's dependency file if provided:

```bash
pip install -r requirements.txt
```

If the project is being run on a machine with a compatible NVIDIA GPU, ensure that the installed PyTorch version supports the available CUDA environment.

The project automatically selects CUDA when available and otherwise uses the CPU.

---

## 5. Evaluation Dataset

The evaluation dataset is located at:

```text
data/evaluation_dataset/
```

The dataset contains:

* 20 ImageNet classes
* 5 images per class
* 100 images total

The images are processed using the ImageNet preprocessing pipeline:

```text
Resize → 256
Center Crop → 224 × 224
ImageNet Normalization
```

---

## 6. Models

The project evaluates two pre-trained ImageNet models:

### ResNet-50

The baseline Top-1 accuracy obtained on the evaluation dataset is:

```text
87.00%
```

### InceptionV3

The baseline Top-1 accuracy obtained on the evaluation dataset is:

```text
75.00%
```

These baseline results are used as the reference point for the robustness experiments.

---

## 7. Weight Perturbation Experiment

The main experiment investigates how model performance changes when a percentage of the learned weights is randomly set to zero.

Three perturbation levels are evaluated:

```text
5%
10%
15%
```

The experiment is performed for both ResNet-50 and InceptionV3.

The general workflow is:

```text
Pre-trained model
       ↓
Baseline evaluation
       ↓
Random weight perturbation
       ↓
5% / 10% / 15%
       ↓
Evaluate perturbed model
       ↓
Compare with baseline
```

The models are not retrained after perturbation. The purpose is to measure their robustness to the introduced parameter damage.

---

## 8. Generated Analysis

The project produces several visual analyses.

### Accuracy vs. Perturbation

Located at:

```text
figures/accuracy_vs_perturbation.png
```

This visualization shows how model accuracy changes as the percentage of perturbed weights increases.

### Layer Sensitivity

Located at:

```text
figures/layer_sensitivity.png
```

This visualization analyzes the sensitivity of different model layers to weight perturbation.

### Robustness Comparison

Located at:

```text
figures/robustness_comparison.png
```

This compares the robustness behavior of ResNet-50 and InceptionV3 under the tested perturbation levels.

### Prediction Comparison

Located at:

```text
figures/prediction_comparisons/
```

The directory contains before-and-after prediction visualizations for the evaluated models:

```text
resnet50_before_after.png
inceptionv3_before_after.png
```

These visualizations allow individual predictions to be compared before and after perturbation.

---

## 9. Numerical Results

The numerical experiment results are stored as CSV files in the project's results/analysis directory.

These files contain the data used to generate the analysis and visualization outputs.

The CSV files can be opened using spreadsheet software or processed programmatically for further analysis.

---

## 10. Reproducibility Test

The project includes a reproducibility test to verify that the experiment can be repeated using the same configuration.

From the project root, with the virtual environment activated, run:

```bash
python src/evaluation/test_reproducibility.py
```

The test should be executed using the same experimental configuration used for the original results.

This test helps verify that the project produces consistent experimental behavior rather than relying on a single execution.

---

## 11. Launching the Streamlit Dashboard

The project includes an interactive Streamlit interface for exploring the analysis results.

From the project root, activate the virtual environment and run:

```bash
streamlit run app/app.py
```

Streamlit will start the application and provide a local URL in the terminal, normally similar to:

```text
http://localhost:8501
```

Open the displayed address in a web browser.

The main application is:

```text
app/app.py
```

The analysis dashboard page is:

```text
app/pages/Analysis_Dashboard.py
```

---

## 12. Using the Dashboard

After launching the Streamlit application, use the available dashboard navigation to explore the project results.

The dashboard is intended to provide an interactive view of the robustness analysis, including:

* Model information
* Baseline performance
* Perturbation results
* Accuracy analysis
* Layer sensitivity
* Robustness comparison
* Prediction visualizations
* Experiment results

The dashboard provides a convenient alternative to manually opening the generated PNG and CSV files.

---

## 13. Recommended Demonstration Workflow

For a complete demonstration, follow this sequence:

1. Show the evaluation dataset.
2. Explain the ImageNet preprocessing pipeline.
3. Introduce ResNet-50 and InceptionV3.
4. Show the baseline accuracy of each model.
5. Explain the weight perturbation procedure.
6. Explain the 5%, 10%, and 15% perturbation levels.
7. Show the accuracy-vs-perturbation analysis.
8. Show the layer sensitivity analysis.
9. Show the robustness comparison.
10. Show the before-and-after prediction visualizations.
11. Open the CSV result files to inspect the underlying numerical results.
12. Run the reproducibility test.
13. Launch the Streamlit dashboard.
14. Explore the results interactively through the dashboard.

---

## 14. Main Research Question

The project is designed to answer the following question:

> How does randomly removing learned neural-network weights affect the classification performance and robustness of different pre-trained CNN architectures?

The experiments provide quantitative and qualitative evidence through accuracy measurements, layer-level analysis, model comparisons, and prediction visualizations.

---

## 15. Important Notes

* The project evaluates a controlled subset of ImageNet rather than the complete ImageNet-1K dataset.
* The evaluation dataset contains 100 images across 20 classes.
* Perturbation levels used in the main experiment are 5%, 10%, and 15%.
* The models are pre-trained ImageNet models.
* Weight perturbation is performed by randomly setting selected weights to zero.
* The models are evaluated after perturbation rather than retrained.
* Results are stored both numerically as CSV files and visually as PNG files.
* The Streamlit dashboard provides an interactive interface for exploring the generated analysis.
