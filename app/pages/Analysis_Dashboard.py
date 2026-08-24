import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.evaluation.reproducibility import (
    run_single_experiment,
    compare_runs,
    save_verification_result
)


# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------
# Custom Theme
# ---------------------------------

st.markdown(
"""
<style>

.main {
    background-color:#F8FAFC;
}


h1 {
    color:#0F172A;
}


h2, h3 {
    color:#1E3A8A;
}


/* Cards */

.card {

    background-color:white;

    border-radius:15px;

    padding:25px;

    height:150px;

    border:1px solid #E2E8F0;

    box-shadow:0px 4px 12px rgba(0,0,0,0.08);

}


.card-title {

    color:#1E3A8A;

    font-size:20px;

    font-weight:700;

}


.card-text {

    color:#475569;

    margin-top:10px;

}


</style>
""",
unsafe_allow_html=True
)



# ---------------------------------
# Load Results
# ---------------------------------

results = pd.read_csv(
    "results/analysis/robustness_analysis_results.csv"
)



# ---------------------------------
# Header
# ---------------------------------

st.title("📊 Robustness Analysis Dashboard")

st.write(
    "Interactive visualization of neural network robustness experiments."
)



# ---------------------------------
# Summary Cards
# ---------------------------------

c1,c2,c3 = st.columns(3)


with c1:

    st.metric(
        "Models Evaluated",
        "2"
    )


with c2:

    st.metric(
        "Perturbation Levels",
        "3"
    )


with c3:

    st.metric(
        "Total Experiments",
        len(results)
    )



st.divider()



# ---------------------------------
# Card Menu
# ---------------------------------

st.subheader("📈 Analysis Visualization")


col1,col2 = st.columns(2)


with col1:

    robustness_btn = st.button(
        "🏆 Model Robustness Comparison",
        use_container_width=True
    )


with col2:

    accuracy_btn = st.button(
        "📉 Accuracy vs Perturbation",
        use_container_width=True
    )



col3, col4 = st.columns(2)

with col3:

    layer_btn = st.button(
        "🧠 Layer Sensitivity",
        use_container_width=True
    )

with col4:

    table_btn = st.button(
        "📋 Experimental Results",
        use_container_width=True
    )


# ---------------------------------
# Fifth Card
# ---------------------------------

col5, col6 = st.columns(2)

with col5:

    reproducibility_btn = st.button(
        "🔬 Reproducibility Verification",
        use_container_width=True
    )


st.divider()



# ---------------------------------
# Display Selected Content
# ---------------------------------

if "selected_view" not in st.session_state:

    st.session_state.selected_view = None



if robustness_btn:

    st.session_state.selected_view = "robustness"


if accuracy_btn:

    st.session_state.selected_view = "accuracy"


if layer_btn:

    st.session_state.selected_view = "layer"


if table_btn:

    st.session_state.selected_view = "table"

if robustness_btn:

    st.session_state.selected_view = "robustness"


if accuracy_btn:

    st.session_state.selected_view = "accuracy"


if layer_btn:

    st.session_state.selected_view = "layer"


if table_btn:

    st.session_state.selected_view = "table"


if reproducibility_btn:

    st.session_state.selected_view = "reproducibility"



if st.session_state.selected_view == "robustness":

    st.subheader(
        "🏆 Model Robustness Comparison"
    )

    st.image(
        "figures/robustness_comparison.png",
        use_container_width=True
    )



elif st.session_state.selected_view == "accuracy":

    st.subheader(
        "📉 Accuracy vs Perturbation"
    )

    st.image(
        "figures/accuracy_vs_perturbation.png",
        use_container_width=True
    )



elif st.session_state.selected_view == "layer":

    st.subheader(
        "🧠 Layer Sensitivity"
    )

    st.image(
        "figures/layer_sensitivity.png",
        use_container_width=True
    )



elif st.session_state.selected_view == "table":

    st.subheader(
        "📋 Experimental Results"
    )

    st.dataframe(
        results,
        use_container_width=True
    )


# ---------------------------------
# Reproducibility Verification
# ---------------------------------

elif st.session_state.selected_view == "reproducibility":

    st.subheader(
        "🔬 Interactive Reproducibility Verification"
    )

    st.write(
        "Select an experimental configuration and "
        "run it twice using the same random seed. "
        "ROBO-Test will compare both runs and save "
        "the verification result as a TXT file."
    )

    st.divider()

    # ---------------------------------
    # Configuration
    # ---------------------------------

    st.markdown("### ⚙ Experiment Configuration")

    r_col1, r_col2 = st.columns(2)

    with r_col1:

        repro_model = st.selectbox(
            "🧠 Model",
            [
                "ResNet-50",
                "InceptionV3"
            ],
            key="repro_model"
        )

        repro_layer = st.selectbox(
            "📍 Layer Region",
            [
                "Early",
                "Middle",
                "Final"
            ],
            key="repro_layer"
        )

    with r_col2:

        repro_perturbation = st.selectbox(
            "⚙ Perturbation",
            [
                5,
                10,
                15
            ],
            format_func=lambda x: f"{x}%",
            key="repro_perturbation"
        )

        repro_seed = st.number_input(
            "🎲 Random Seed",
            min_value=0,
            value=42,
            step=1,
            key="repro_seed"
        )

    # ---------------------------------
    # Dataset Images
    # ---------------------------------

    st.markdown("### 🖼 Evaluation Image")

    dataset_root = Path(
        "data/evaluation_dataset"
    )

    available_images = []

    if dataset_root.exists():

        for image_path in sorted(
            dataset_root.rglob("*")
        ):

            if image_path.suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png"
            ]:

                available_images.append(
                    image_path
                )

    if not available_images:

        st.error(
            "No evaluation dataset images were found."
        )

    else:

        image_options = {
            str(
                image.relative_to(dataset_root)
            ): image
            for image in available_images
        }

        selected_image_name = st.selectbox(
            "Select Image",
            list(image_options.keys()),
            key="repro_image"
        )

        selected_image = image_options[
            selected_image_name
        ]

        st.caption(
            f"Selected image: {selected_image}"
        )

        # ---------------------------------
        # Run Verification
        # ---------------------------------

        verify_button = st.button(
            "▶ Verify Reproducibility",
            use_container_width=True,
            type="primary"
        )

        if verify_button:

            with st.spinner(
                "Running the experiment twice..."
            ):

                run1 = run_single_experiment(
                    model_name=repro_model,
                    layer_region=repro_layer,
                    perturbation_percentage=repro_perturbation,
                    seed=repro_seed,
                    image_path=selected_image
                )

                run2 = run_single_experiment(
                    model_name=repro_model,
                    layer_region=repro_layer,
                    perturbation_percentage=repro_perturbation,
                    seed=repro_seed,
                    image_path=selected_image
                )

                comparison = compare_runs(
                    run1,
                    run2
                )

                output_file = (
                    save_verification_result(
                        model_name=repro_model,
                        layer_region=repro_layer,
                        perturbation_percentage=repro_perturbation,
                        seed=repro_seed,
                        image_path=selected_image,
                        run1=run1,
                        run2=run2,
                        comparison=comparison
                    )
                )

            # ---------------------------------
            # Result
            # ---------------------------------

            st.divider()

            st.subheader(
                "📊 Reproducibility Result"
            )

            if comparison["reproducible"]:

                st.success(
                    "🟢 REPRODUCIBLE — "
                    "Both experiments produced "
                    "identical results."
                )

            else:

                st.error(
                    "🔴 NOT REPRODUCIBLE — "
                    "The two experiments produced "
                    "different results."
                )

            # ---------------------------------
            # Configuration Summary
            # ---------------------------------

            st.markdown(
                "### Experiment Configuration"
            )

            st.write(
                f"**Model:** {repro_model}"
            )

            st.write(
                f"**Layer:** {repro_layer}"
            )

            st.write(
                f"**Perturbation:** "
                f"{repro_perturbation}%"
            )

            st.write(
                f"**Seed:** {repro_seed}"
            )

            st.write(
                f"**Image:** "
                f"{selected_image.name}"
            )

            # ---------------------------------
            # Run Comparison
            # ---------------------------------

            st.markdown(
                "### 🔄 Run Comparison"
            )

            comparison_col1, comparison_col2 = (
                st.columns(2)
            )

            with comparison_col1:

                st.markdown(
                    "#### Experiment 1"
                )

                st.write(
                    "**Before:** "
                    f"{run1['before']['class_name'].title()}"
                )

                st.write(
                    "**After:** "
                    f"{run1['after']['class_name'].title()}"
                )

                st.metric(
                    "Before Confidence",
                    f"{run1['before']['confidence'] * 100:.2f}%"
                )

                st.metric(
                    "After Confidence",
                    f"{run1['after']['confidence'] * 100:.2f}%"
                )

                st.write(
                    f"**Modified Weights:** "
                    f"{run1['modified_count']:,}"
                )

            with comparison_col2:

                st.markdown(
                    "#### Experiment 2"
                )

                st.write(
                    "**Before:** "
                    f"{run2['before']['class_name'].title()}"
                )

                st.write(
                    "**After:** "
                    f"{run2['after']['class_name'].title()}"
                )

                st.metric(
                    "Before Confidence",
                    f"{run2['before']['confidence'] * 100:.2f}%"
                )

                st.metric(
                    "After Confidence",
                    f"{run2['after']['confidence'] * 100:.2f}%"
                )

                st.write(
                    f"**Modified Weights:** "
                    f"{run2['modified_count']:,}"
                )

            # ---------------------------------
            # Verification Checks
            # ---------------------------------

            st.markdown(
                "### ✅ Verification Checks"
            )

            checks = pd.DataFrame({

                "Verification": [
                    "Before Prediction",
                    "Before Confidence",
                    "After Prediction",
                    "After Confidence",
                    "Modified Weight Count",
                    "Perturbed Model Hash"
                ],

                "Result": [
                    "PASS"
                    if comparison[
                        "before_prediction_match"
                    ]
                    else "FAIL",

                    "PASS"
                    if comparison[
                        "before_confidence_match"
                    ]
                    else "FAIL",

                    "PASS"
                    if comparison[
                        "after_prediction_match"
                    ]
                    else "FAIL",

                    "PASS"
                    if comparison[
                        "after_confidence_match"
                    ]
                    else "FAIL",

                    "PASS"
                    if comparison[
                        "modified_count_match"
                    ]
                    else "FAIL",

                    "PASS"
                    if comparison[
                        "model_hash_match"
                    ]
                    else "FAIL"
                ]

            })

            st.dataframe(
                checks,
                use_container_width=True,
                hide_index=True
            )

            # ---------------------------------
            # Saved File
            # ---------------------------------

            st.success(
                f"Verification saved to: "
                f"`{output_file}`"
            )


else:

    st.info(
        "Select an analysis card above to view results."
    )