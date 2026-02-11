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
    confusion_matrix
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
# Premium Styling
# --------------------------------------------------
st.markdown("""
<style>

/* App background */
.stApp {
    background-color: #eef2f8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
    padding-top: 30px;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Header */
.header-box {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    padding: 35px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
}

/* Student Info */
.student-box {
    background: white;
    padding: 18px 25px;
    border-radius: 15px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
    margin-bottom: 30px;
}

/* Content cards */
.section-box {
    background: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.06);
    margin-bottom: 30px;
}

/* Metric spacing */
[data-testid="metric-container"] {
    background-color: #f8faff;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header Section
# --------------------------------------------------
st.markdown("""
<div class="header-box">
    <h1>💼 Adult Income Classification Dashboard</h1>
    <p>Machine Learning Model Evaluation System</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="student-box">
    <b>👤 Dinesh B M</b> &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; 🆔 2025AA05364
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.markdown("## 🔎 Select Model")

model_options = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree Classifier": "decision_tree",
    "K-Nearest Neighbor Classifier": "knn",
    "Naive Bayes Classifier (Gaussian)": "naive_bayes",
    "Ensemble Model - Random Forest": "random_forest",
    "Ensemble Model - XGBoost": "xgboost"
}

selected_model_display = st.sidebar.selectbox(
    "",
    list(model_options.keys())
)

model_file_name = model_options[selected_model_display]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Sample Dataset")

try:
    raw_df = pd.read_csv("data/adult.csv")
    sample_df = raw_df.sample(n=2000, random_state=42)

    st.sidebar.download_button(
        "Download Sample (2000 rows)",
        sample_df.to_csv(index=False),
        "adult_sample_2000.csv",
        "text/csv"
    )
except:
    st.sidebar.warning("Sample dataset not found.")

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

# --------------------------------------------------
# Main Logic
# --------------------------------------------------
if uploaded_file is None:
    st.info("⬅️ Upload a dataset from the sidebar to begin evaluation.")
else:
    data = pd.read_csv(uploaded_file)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("📄 Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if "income" not in data.columns:
        st.error("❌ CSV must contain the 'income' column.")
    else:
        model_path = f"models/{model_file_name}.pkl"
        model = joblib.load(model_path)

        X_test = data.drop("income", axis=1)
        y_test = data["income"].map({"<=50K": 0, ">50K": 1})

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # --------------------------------------------------
        # Performance Overview
        # --------------------------------------------------
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader(f"📊 Performance Overview : {selected_model_display}")

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        row1 = st.columns(3)
        row2 = st.columns(3)

        row1[0].metric("Accuracy", f"{acc:.3f}")
        row1[1].metric("AUC", f"{auc:.3f}")
        row1[2].metric("Precision", f"{prec:.3f}")

        row2[0].metric("Recall", f"{rec:.3f}")
        row2[1].metric("F1 Score", f"{f1:.3f}")
        row2[2].metric("MCC", f"{mcc:.3f}")

        st.markdown('</div>', unsafe_allow_html=True)

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
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
            linewidths=0.5,
            linecolor="gray",
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(selected_model_display)

        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.success("✅ Evaluation Completed Successfully")