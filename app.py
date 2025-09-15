import streamlit as st
import pandas as pd
from datasets_search import (
    load_openml_dataset,
    load_huggingface_dataset,
    list_openml_datasets,
    list_huggingface_datasets,
)
from preprocessing import preprocess_data
from models import train_and_evaluate_automl
from utils import (
    plot_correlation_matrix,
    plot_target_distribution,
    plot_top_feature_distributions,
    eda_summary,
)

st.set_page_config(page_title="AutoML Dashboard", layout="wide")

st.title("🧠 AutoML Dashboard: From Data to Decisions")

# Dataset selection
st.sidebar.header("📂 Dataset Options")
dataset_source = st.sidebar.radio("Choose dataset source:", ["Upload CSV", "OpenML", "Hugging Face"])

df = None

if dataset_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

elif dataset_source == "OpenML":
    st.sidebar.caption("Search and pick an OpenML dataset with a short description.")
    datasets_df = list_openml_datasets(limit=300)
    options = [f"{row.did} — {row.description}" for _, row in datasets_df.iterrows()] if not datasets_df.empty else []
    selected = st.sidebar.selectbox("OpenML dataset (ID — description)", options)
    if selected:
        openml_id = selected.split(" — ")[0]
        if st.sidebar.button("Load from OpenML"):
            df = load_openml_dataset(openml_id)

elif dataset_source == "Hugging Face":
    st.sidebar.caption("Browse popular Hugging Face datasets.")
    hf_df = list_huggingface_datasets(limit=200)
    options = [f"{row.name} — {row.description}" for _, row in hf_df.iterrows()] if not hf_df.empty else []
    selected = st.sidebar.selectbox("Hugging Face dataset (name — description)", options)
    if selected:
        hf_name = selected.split(" — ")[0]
        if st.sidebar.button("Load from Hugging Face"):
            df = load_huggingface_dataset(hf_name)

if df is not None and not df.empty and "error" not in df.columns:
    st.write("### 📊 Dataset Preview")
    st.dataframe(df.head(50))
    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    target_col = st.selectbox("🎯 Select target column (the value we want to predict)", df.columns)
    if target_col:
        # EDA
        info, missing_by_col = eda_summary(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{info['rows']}")
        c2.metric("Columns", f"{info['columns']}")
        c3.metric("Missing (%)", f"{info['missing_pct']:.1f}%")

        with st.expander("See missing values by column"):
            st.dataframe(missing_by_col.to_frame("missing_fraction"))

        st.write("### 🔍 Distributions")
        st.pyplot(plot_top_feature_distributions(df))
        st.write("### 🔎 Correlation Matrix")
        st.pyplot(plot_correlation_matrix(df))

        preprocess, task_type, X_train, X_test, y_train, y_test = preprocess_data(df, target_col)
        st.write("### 🤖 AutoML Training")
        st.caption("We try a few well-rounded models and pick the best based on validation performance.")
        automl = train_and_evaluate_automl(task_type, preprocess, X_train, X_test, y_train, y_test)

        st.write("#### Leaderboard (higher is better for accuracy, lower is better for RMSE)")
        lb_df = pd.DataFrame(automl["leaderboard"]).T
        st.dataframe(lb_df)

        st.success(f"Selected model: {automl['best_model_name']}")
        st.info(automl["reasoning"])

        st.write("### 🧩 Feature Influence")
        if automl["feature_importance"]:
            top_k = automl["feature_importance"][:20]
            fi_df = pd.DataFrame(top_k, columns=["feature", "importance"]) 
            st.bar_chart(fi_df.set_index("feature"))
        else:
            st.caption("Permutation importance not available for this dataset/model.")

        # Predictions + download
        st.write("### 📥 Predictions")
        best_model = automl["best_model"]
        if best_model is not None:
            preds = best_model.predict(X_test)
            pred_df = pd.DataFrame({"y_true": y_test})
            pred_df["y_pred"] = preds
            st.dataframe(pred_df.head(100))

            # Simple visualization
            if task_type == "classification":
                st.caption("Prediction counts")
                st.bar_chart(pred_df["y_pred"].value_counts())
            else:
                st.caption("Predicted vs Actual")
                st.scatter_chart(pred_df.rename(columns={"y_true": "Actual", "y_pred": "Predicted"}))

            # Download predictions CSV
            csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download predictions as CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )
        else:
            st.warning("No model trained.")
elif df is not None and "error" in df.columns:
    st.error(f"Dataset load error: {df.loc[0, 'error']}")
