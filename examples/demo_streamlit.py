# examples/demo_streamlit.py
import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from explainy.core import Explainer
from explainy.visual import plot_global_importance

st.title("Explainy — quick explainability demo")

@st.cache_data
def load_data():
    data = load_breast_cancer(as_frame=True)
    X = data.frame[data.feature_names]
    y = data.target
    return X, y, data

X, y, data_obj = load_data()
st.sidebar.write("Dataset: Breast cancer (sklearn)")
st.write("Sample data")
st.dataframe(X.head())

# Initialize session state keys
if "model" not in st.session_state:
    st.session_state.model = None
if "expl" not in st.session_state:
    st.session_state.expl = None

# --- Train model ---
if st.sidebar.button("Train model"):
    model = RandomForestClassifier(n_estimators=50, random_state=0)
    model.fit(X, y)
    expl = Explainer(model, X, feature_names=list(X.columns), predict_proba=True)

    st.session_state.model = model
    st.session_state.expl = expl

    st.success("✅ Model trained and Explainer initialized")

# --- Show explainability parts if model is trained ---
if st.session_state.expl is not None:
    expl = st.session_state.expl

    # Global importance
    st.header("Global importance")
    g = expl.global_feature_importance()
    fig = plot_global_importance(g, top_n=10)
    st.pyplot(fig)

    # Local explanation
    st.header("Local explanation")
    row_idx = st.slider("Row index to explain", 0, len(X)-1, 0)
    x_row = X.iloc[row_idx]
    expl_local = expl.explain_local(x_row)
    st.json(expl_local)

    # Counterfactual demo
    st.header("Simple Counterfactual Demo")
    st.write("Tries to find a nearby example where the model flips prediction.")

    # Let user select desired target class
    target_class = st.radio(
        "Desired outcome:",
        options=[0, 1],
        format_func=lambda x: "Benign (0)" if x == 0 else "Malignant (1)",
        horizontal=True
    )

    def target_fn(pred):
        if pred.ndim == 2:  # predict_proba case
            return pred[0, target_class] > 0.5
        return pred[0] == target_class

    cf = expl.simple_counterfactual(x_row, target_fn=target_fn)

    if cf is not None:
        st.subheader("Counterfactual found")
        st.write(f"Modified example that predicts **{ 'Malignant' if target_class==1 else 'Benign' }**:")

        diff = pd.DataFrame({"Original": x_row, "Counterfactual": cf})
        diff["Changed?"] = diff["Original"] != diff["Counterfactual"]
        st.dataframe(diff)
    else:
        st.warning("No counterfactual found with the simple greedy search.")
