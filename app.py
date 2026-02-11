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
    layout="wide",
    page_icon="💼"
)

# --------------------------------------------------
# Custom Styling (Clean Modern Look)
# --------------------------------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 18px;
        color: #6c757d;
        margin-top: 0px;
    }
    .student-box {
        background-color: #f1f3f6;
        padding: 10px 15px;
        border-radius: 10px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header Section
# --------------------------------------------------
st.markdown('<div class="main-title">💼 Adult Income Classification Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning Model Evaluation System</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="student-box"><b>👤 Dinesh B M</b> | 🆔 2025AA05364</div>',
    unsafe_allow_html=True
)

st.markdown("""
This application allows evaluation of multiple machine learning models trained
on the **Adult Census Income dataset**.  
Upload a raw CSV file or download a sample dataset to test the models.
""")

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("⚙️ Model Controls")

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

st.sidebar.markdown("---")

# Sample Download
st.sidebar.subheader("📥 Sample Dataset")

try:
    raw_df = pd.read_csv("data/adult.csv")
    sample_df = raw_df.sample(n=2000, random_state=42)

    st.sidebar.download_button(
        label="Download Sample (2000 rows)",
        data=sample_df.to_csv(index=False),
        file_name="adult_sample_2000.csv",
        mime="text/csv"
    )
except:
    st.sidebar.warning("Sample dataset not found.")

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

# --------------------------------------------------
# Main Logic
# --------------------------------------------------
if uploaded_file is None:
    st.info("⬅️ Upload a dataset from the sidebar to begin evaluation.")
else:
    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    if "income" not in data.columns:
        st.error("❌ CSV must contain the 'income' column.")
    else:
        model_path = f"models/{model_name.lower().replace(' ', '_')}.pkl"
        model = joblib.load(model_path)

        X_test = data.drop("income", axis=1)
        y_test = data["income"].map({"<=50K": 0, ">50K": 1})

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # --------------------------------------------------
        # Metrics Section
        # --------------------------------------------------
        st.divider()
        st.subheader("📊 Performance Overview")

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("Accuracy", f"{acc:.3f}")
        col2.metric("AUC", f"{auc:.3f}")
        col3.metric("Precision", f"{prec:.3f}")
        col4.metric("Recall", f"{rec:.3f}")
        col5.metric("F1 Score", f"{f1:.3f}")
        col6.metric("MCC", f"{mcc:.3f}")

        # --------------------------------------------------
        # Classification Summary
        # --------------------------------------------------
        st.divider()
        st.subheader("📈 Class-wise Performance")

        report = classification_report(y_test, y_pred, output_dict=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### ≤50K")
            st.metric("Precision", f"{report['0']['precision']:.3f}")
            st.metric("Recall", f"{report['0']['recall']:.3f}")
            st.metric("F1-score", f"{report['0']['f1-score']:.3f}")

        with c2:
            st.markdown("### >50K")
            st.metric("Precision", f"{report['1']['precision']:.3f}")
            st.metric("Recall", f"{report['1']['recall']:.3f}")
            st.metric("F1-score", f"{report['1']['f1-score']:.3f}")

        # --------------------------------------------------
        # Confusion Matrix
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
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"{model_name} Confusion Matrix")

        st.pyplot(fig)

        st.success("✅ Evaluation Completed Successfully")