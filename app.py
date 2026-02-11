import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import os

# Helper Functions
@st.cache_data
def load_sample_data(file_path, sample_size=2000):
    """Load and sample data from a CSV file."""
    try:
        raw_df = pd.read_csv(file_path)
        return raw_df.sample(n=sample_size, random_state=42)
    except Exception as e:
        st.sidebar.error(f"Error loading sample data: {str(e)}")
        return None

def load_model(model_name):
    """Load a trained model from file."""
    model_path = f"models/{model_name.lower().replace(' ', '_')}.pkl"
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def process_uploaded_file(uploaded_file):
    """Process the uploaded CSV file."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        return None

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate various classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and return a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["≤50K", ">50K"],
        yticklabels=["≤50K", ">50K"],
        cbar=True,
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix – {model_name}")
    return fig

# Page Configuration
st.set_page_config(page_title="Adult Income Classification", layout="wide")

# User Information (Top Right)
st.markdown(
    """
    <style>
    .user-info {
        position: fixed;
        top: 0;
        right: 0;
        padding: 10px;
        background-color: #f0f2f6;
        border-bottom-left-radius: 10px;
        z-index: 1000;
    }
    </style>
    <div class="user-info">
        <p style="margin: 0; font-size: 0.8em;">Dinesh B M (ID: 2025AA05364)</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Title & Description
st.markdown("## 💼 Adult Income Classification App")
st.markdown(
    """
    This application demonstrates **multiple machine learning classification models**
    trained on the **Adult Census Income dataset**.
    
    You can **upload a raw CSV file** or **download a sample dataset**
    to evaluate model performance.
    """
)

st.divider()

# Sidebar Controls
st.sidebar.header("🔧 Controls")
model_name = st.sidebar.selectbox(
    "Select Classification Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# Sample Data Download
st.sidebar.subheader("📥 Sample Data (Raw)")
sample_df = load_sample_data("data/adult.csv")
if sample_df is not None:
    st.sidebar.download_button(
        label="⬇️ Download Sample Data (2000 rows)",
        data=sample_df.to_csv(index=False),
        file_name="adult_sample_2000.csv",
        mime="text/csv"
    )

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

st.sidebar.info(
    "📌 CSV must contain all feature columns\n"
    "and the **income** target column."
)

# Main Logic
if uploaded_file is None:
    st.warning("⬅️ Upload a CSV file or download sample data to continue.")
else:
    # Display a progress bar while processing the file
    with st.spinner("Processing uploaded file..."):
        data = process_uploaded_file(uploaded_file)
    
    if data is None:
        st.error("Failed to process the uploaded file. Please check the file format and try again.")
    else:
        # Dataset Preview
        st.subheader("📄 Uploaded Dataset Preview")
        st.dataframe(data.head(), use_container_width=True, height=220)

        if "income" not in data.columns:
            st.error("❌ Uploaded CSV must contain the 'income' column.")
        else:
            # Load trained pipeline model
            with st.spinner(f"Loading {model_name} model..."):
                model = load_model(model_name)
            
            if model is None:
                st.error(f"Failed to load the {model_name} model. Please try another model or contact support.")
            else:
                # Prepare data
                X_test = data.drop("income", axis=1)
                y_test = data["income"].map({"<=50K": 0, ">50K": 1})

                # Predictions
                with st.spinner("Making predictions..."):
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]

                # Metrics Calculation and Display
                metrics = calculate_metrics(y_test, y_pred, y_prob)
                
                st.divider()
                st.subheader("📊 Model Performance Metrics")

                c1, c2, c3 = st.columns(3)
                c4, c5, c6 = st.columns(3)

                c1.metric("Model", model_name)
                c2.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                c3.metric("AUC", f"{metrics['auc']:.3f}")
                c4.metric("Precision", f"{metrics['precision']:.3f}")
                c5.metric("Recall", f"{metrics['recall']:.3f}")
                c6.metric("F1 Score", f"{metrics['f1']:.3f}")

                st.metric("Matthews Correlation Coefficient (MCC)", f"{metrics['mcc']:.3f}")

                # Classification Summary
                st.divider()
                st.subheader("📈 Classification Summary")

                report = classification_report(y_test, y_pred, output_dict=True)

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("### ≤50K Class")
                    st.metric("Precision", f"{report['0']['precision']:.3f}")
                    st.metric("Recall", f"{report['0']['recall']:.3f}")
                    st.metric("F1-score", f"{report['0']['f1-score']:.3f}")

                with col_b:
                    st.markdown("### >50K Class")
                    st.metric("Precision", f"{report['1']['precision']:.3f}")
                    st.metric("Recall", f"{report['1']['recall']:.3f}")
                    st.metric("F1-score", f"{report['1']['f1-score']:.3f}")

                with st.expander("📋 Full Classification Report (Table View)"):
                    report_df = pd.DataFrame(report).transpose().round(3)
                    st.dataframe(report_df, use_container_width=True)

                # Confusion Matrix
                st.divider()
                st.subheader("🔢 Confusion Matrix")

                cm_fig = plot_confusion_matrix(y_test, y_pred, model_name)
                st.pyplot(cm_fig)

                st.success("✅ Evaluation completed successfully!")
