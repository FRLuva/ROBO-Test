import sys
from pathlib import Path

# ---------------------------------------
# Add project root to Python path
# ---------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
from PIL import Image
import pandas as pd
from pathlib import Path

from src.perturbation.layer_weight_perturbation import (
    apply_layer_weight_perturbation
)

from src.evaluation.inference import (
    load_model,
    predict_image
)

# ---------------------------------------
# Page Configuration
# ---------------------------------------
st.set_page_config(
    page_title="ROBO-Test",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

.main {
    background-color:#F5F7FA;
}


/* Sidebar background */
section[data-testid="stSidebar"]{
    background-color:#0F172A;
}


/* Sidebar normal text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label{
    color:white !important;
}


/* -------------------------
   Selectbox Fix
------------------------- */


/* Dropdown selected box */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div{

    background-color:white !important;

}


/* Selected value text */
section[data-testid="stSidebar"] div[data-baseweb="select"] div{

    color:#0F172A !important;

}


/* Dropdown arrow */
section[data-testid="stSidebar"] svg{

    fill:#0F172A !important;

}


/* -------------------------
   File uploader Fix
------------------------- */

section[data-testid="stSidebar"] [data-testid="stFileUploader"]{

    color:white !important;

}


section[data-testid="stSidebar"] [data-testid="stFileUploader"] small{

    color:white !important;

}

/* File uploader button */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {

    background-color:#1D4ED8 !important;
    color:white !important;
    border-radius:8px !important;
    border:none !important;
    font-weight:bold !important;

}


section[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {

    background-color:#2563EB !important;

}


/* -------------------------
   Button
------------------------- */

.stButton > button{

    width:100%;
    background-color:#1D4ED8;
    color:white !important;
    border-radius:10px;
    height:45px;
    border:none;
    font-size:16px;
    font-weight:bold;

}


.stButton > button:hover{

    background-color:#2563EB;

}


</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# Cache Models
# ---------------------------------------
@st.cache_resource
def get_model(model_name):
    return load_model(model_name)

LAYER_MAPPING = {

    "ResNet-50": {
        "Early": "conv1",
        "Middle": "layer2",
        "Final": "fc"
    },

    "InceptionV3": {
        "Early": "Conv2d",
        "Middle": "Mixed_6",
        "Final": "fc"
    }

}

@st.cache_data
def load_robustness_results():

    csv_path = Path(
        "results/analysis/robustness_analysis_results.csv"
    )

    return pd.read_csv(csv_path)

robustness_df = load_robustness_results()

# ---------------------------------------
# Sidebar
# ---------------------------------------
st.sidebar.title("🤖 ROBO-Test")
st.sidebar.write("Robustness Analysis of Neural Networks")
st.sidebar.divider()

with st.sidebar.form("analysis_form"):

    model = st.selectbox(
        "🧠 Select Model",
        ["ResNet-50", "InceptionV3"]
    )

    perturbation = st.selectbox(
        "⚙ Select Perturbation",
        ["5%", "10%", "15%"]
    )

    layer_region = st.selectbox(
        "📍 Layer Region",
        ["Early", "Middle", "Final"]
    )  

    uploaded_file = st.file_uploader(
        "🖼 Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    run = st.form_submit_button(
        "▶ Run Analysis",
        use_container_width=True
    )

st.sidebar.divider()

# ---------------------------------------
# Run Analysis
# ---------------------------------------

before_result = None
after_result = None
modified_count = None

if run and uploaded_file is not None:

    with st.spinner("Running robustness analysis..."):

        model_name = model.replace("-", "")

        original_model = get_model(model_name)

        # ---------- BEFORE ----------
        uploaded_file.seek(0)

        before_result = predict_image(
            original_model,
            uploaded_file
        )

        # ---------- PERTURB ----------
        layer_keyword = LAYER_MAPPING[model][layer_region]

        perturbed_model, modified_count = apply_layer_weight_perturbation(
            original_model,
            layer_keyword,
            int(perturbation.replace("%","")),
            seed=42
        )

        # ---------- AFTER ----------
        uploaded_file.seek(0)

        after_result = predict_image(
            perturbed_model,
            uploaded_file
        )

        # ---------- DEBUG ----------
        uploaded_file.seek(0)

        check_result = predict_image(
            original_model,
            uploaded_file
        )

        print("\n==============================")
        print("Before        :", before_result)
        print("After         :", after_result)
        print("Original Again:", check_result)
        print("==============================")

    st.sidebar.success("✅ Analysis Complete")

elif run:

    st.sidebar.error("Please upload an image.")

else:

    st.sidebar.info("Ready")

# ---------------------------------------
# Main Page
# ---------------------------------------
st.title("ROBO-Test")
st.subheader("Interactive Robustness Analysis Dashboard")

st.divider()

left, right = st.columns([1,1])

# ---------------------------------------
# Left Column
# ---------------------------------------
with left:

    st.subheader("🖼 Uploaded Image")

    if uploaded_file is not None:

        uploaded_file.seek(0)

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    else:

        st.info("Please upload an image from the sidebar.")

# ---------------------------------------
# Right Column
# ---------------------------------------
with right:

    st.subheader("📊 Prediction Comparison")

    before_col, after_col = st.columns(2)

    # -------------------
    # BEFORE
    # -------------------
    with before_col:

        st.markdown("### Before Perturbation")

        if before_result is not None:

            st.write("**Prediction**")
            st.success(before_result["class_name"].title())

            st.metric(
                "Confidence",
                f"{before_result['confidence']*100:.2f}%"
            )

        else:

            st.metric("Prediction", "-")
            st.metric("Confidence", "-")

# -------------------
# AFTER
# -------------------
with after_col:

    st.markdown("### After Perturbation")

    if after_result is not None:

        st.write("**Prediction**")
        st.success(after_result["class_name"].title())

        st.metric(
            "Confidence",
            f"{after_result['confidence']*100:.2f}%"
        )

    else:

        st.metric("Prediction", "-")
        st.metric("Confidence", "-")

st.divider()

st.subheader("Experiment Information")

st.write(f"**Model:** {model}")
st.write(f"**Layer Region:** {layer_region}")
st.write(f"**Perturbation:** {perturbation}")

if modified_count is not None:
    st.write(f"**Modified Weights:** {modified_count:,}")

st.divider()

st.subheader("Prediction Status")

if before_result is not None and after_result is not None:

    if before_result["class_index"] == after_result["class_index"]:

        st.success("🟢 Prediction Stable")

    else:

        st.error("🔴 Prediction Changed")

st.divider()

st.subheader("Prediction Confidence")

if before_result is not None and after_result is not None:

    before = before_result["confidence"] * 100
    after = after_result["confidence"] * 100

    delta = after - before

    st.metric(
        label="Prediction Confidence",
        value=f"{after:.2f}%",
        delta=f"{delta:.2f}%"
    )

    st.write(f"**Before:** {before:.2f}%")
    st.write(f"**After:** {after:.2f}%")

else:

    st.metric(
        label="Prediction Confidence",
        value="-",
        delta="-"
    )

st.divider()

st.subheader("Research Result")

if run:

    selected = robustness_df[
        (robustness_df["Model"] == model)
        &
        (robustness_df["Region"] == layer_region)
        &
        (
            robustness_df["Perturbation_Level"]
            ==
            int(perturbation.replace("%", ""))
        )
    ]

    if not selected.empty:

        baseline = float(
            selected.iloc[0]["Baseline_Accuracy"]
        )

        perturbed = float(
            selected.iloc[0]["Perturbed_Accuracy"]
        )

        drop = float(
            selected.iloc[0]["Accuracy_Drop"]
        )

        st.metric(
            "Dataset Accuracy",
            f"{perturbed:.1f}%",
            delta=f"-{drop:.1f}%"
        )

        st.caption(
            f"Baseline Accuracy: {baseline:.1f}% "
            f"(100 evaluation images)"
        )