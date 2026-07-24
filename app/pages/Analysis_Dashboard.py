import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd


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



col3,col4 = st.columns(2)


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


else:

    st.info(
        "Select an analysis card above to view results."
    )