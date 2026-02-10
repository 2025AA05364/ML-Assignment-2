import streamlit as st
import pandas as pd
import joblib

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
    
    Upload a **test CSV file**, select a model, and view **clear evaluation metrics**
    and results.
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

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

st.sidebar.info(
    "📌 CSV must contain the same feature columns\n"
    "and the **income** target column."
)

# --------------------------------------------------
# Main Logic
# --------------------------------------------------
if uploaded_file is None:
    st.warning("⬅️ Please upload a test CSV file to continue.")
else:
    # Read CSV
    data = pd.read_csv(uploaded_file)

    # --------------------------------------------------
    # Dataset Preview
    # --------------------------------------------------
    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(
        data.head(),
        use_container_width=True,
        height=220
    )

    if "income" not in data.columns:
        st.error("❌ Uploaded CSV must contain the 'income' column.")
    else:
        # Load selected model
        model_path = f"models/{model_name.lower().replace(' ', '_')}.pkl"
        model = joblib.load(model_path)

        # Prepare test data
        X_test = data.drop("income", axis=1)
        y_test = data["income"].map({"<=50K": 0, ">50K": 1})

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba") else y_pred
        )

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
        # Metrics Display (CARDS)
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
        # Classification Report
        # --------------------------------------------------
        st.divider()
        st.subheader("📈 Classification Report")

        report_df = (
            pd.DataFrame(
                classification_report(
                    y_test,
                    y_pred,
                    output_dict=True
                )
            )
            .transpose()
            .round(3)
        )

        st.dataframe(
            report_df,
            use_container_width=True,
            height=320
        )

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------
        st.divider()
        st.subheader("🔢 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        cm_df = pd.DataFrame(
            cm,
            columns=["Predicted ≤50K", "Predicted >50K"],
            index=["Actual ≤50K", "Actual >50K"]
        )

        st.dataframe(
            cm_df,
            use_container_width=True,
            height=150
        )

        st.success("✅ Evaluation completed successfully!")