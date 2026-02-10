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

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Adult Income Classification",
    layout="wide"
)

# --------------------------------------------------
# Title & Description
# --------------------------------------------------
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

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
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

# --------------------------------------------------
# Sample Data Download
# --------------------------------------------------
st.sidebar.subheader("📥 Sample Data (Raw)")

try:
    raw_df = pd.read_csv("data/adult.csv")
    sample_df = raw_df.sample(n=2000, random_state=42)

    st.sidebar.download_button(
        label="⬇️ Download Sample Data (2000 rows)",
        data=sample_df.to_csv(index=False),
        file_name="adult_sample_2000.csv",
        mime="text/csv"
    )
except Exception:
    st.sidebar.warning("Sample data not found in data/adult.csv")

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

st.sidebar.info(
    "📌 CSV must contain all feature columns\n"
    "and the **income** target column."
)

# --------------------------------------------------
# Main Logic
# --------------------------------------------------
if uploaded_file is None:
    st.warning("⬅️ Upload a CSV file or download sample data to continue.")
else:
    data = pd.read_csv(uploaded_file)

    # Dataset Preview
    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(data.head(), use_container_width=True, height=220)

    if "income" not in data.columns:
        st.error("❌ Uploaded CSV must contain the 'income' column.")
    else:
        # Load trained pipeline model
        model_path = f"models/{model_name.lower().replace(' ', '_')}.pkl"
        model = joblib.load(model_path)

        # Prepare data
        X_test = data.drop("income", axis=1)
        y_test = data["income"].map({"<=50K": 0, ">50K": 1})

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # --------------------------------------------------
        # Metrics Calculation
        # --------------------------------------------------
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # --------------------------------------------------
        # Metrics Display
        # --------------------------------------------------
        st.divider()
        st.subheader("📊 Model Performance Metrics")

        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)

        c1.metric("Model", model_name)
        c2.metric("Accuracy", f"{acc:.3f}")
        c3.metric("AUC", f"{auc:.3f}")
        c4.metric("Precision", f"{prec:.3f}")
        c5.metric("Recall", f"{rec:.3f}")
        c6.metric("F1 Score", f"{f1:.3f}")

        st.metric("Matthews Correlation Coefficient (MCC)", f"{mcc:.3f}")

        # --------------------------------------------------
        # Classification Summary (UX Improved)
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Confusion Matrix (STANDARD HEATMAP)
        # --------------------------------------------------
        st.divider()
        st.subheader("🔢 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

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

        st.pyplot(fig)

        st.success("✅ Evaluation completed successfully!")