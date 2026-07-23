"""
Verify baseline evaluation results.

Checks:
1. Baseline accuracy files exist.
2. Sample prediction files exist.
3. Files contain saved results.
"""

from pathlib import Path


RESULT_FILES = [
    "results/baseline_resnet50.txt",
    "results/baseline_inceptionv3.txt",
    "results/sample_predictions_resnet50.txt",
    "results/sample_predictions_inceptionv3.txt"
]


def verify_results():

    print("=" * 60)
    print("Baseline Result Verification")
    print("=" * 60)


    for file_path in RESULT_FILES:

        path = Path(file_path)

        if path.exists():

            size = path.stat().st_size

            print(
                f"✓ Found: {file_path}"
            )

            print(
                f"  Size: {size} bytes"
            )

        else:

            print(
                f"✗ Missing: {file_path}"
            )


if __name__ == "__main__":

    verify_results()