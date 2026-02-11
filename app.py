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
# Modern Styling
# --------------------------------------------------
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #f4f7fc;
}

/* Header */
.header-box {
    background: linear-gradient(135deg, #1f3c88, #3a7bd5);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 20px;
}

/* Student box */
.student-box {
    background-color: white;
    padding: 15px 20px;
    border-radius: 14px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Content cards */
.section-box {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.06);
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div class="header-box">
    <h1>💼 Adult Income Classification Dashboard</h1>
    <p>Machine Learning Model Evaluation System</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="student-box">
    <b>👤 Dinesh B M</b> &nbsp;&nbsp; | &nbsp;&nbsp; 🆔 2025AA05364
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("🔧 Controls")

model_display_options = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree Classifier": "decision_tree",
    "K-Nearest Neighbor Classifier": "knn",
    "Naive Bayes Classifier (Gaussian)": "naive_bayes",
    "Ensemble Model - Random Forest": "random_forest",
    "Ensemble Model - XGBoost": "xgboost"
}

model_name_display = st.sidebar.selectbox(
    "Select Classification Model",
    list(model_display_options.keys())
)

model_file_name = model_display_options[model_name_display]

# --------------------------------------------------
# Sample Data Section
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
except:
    st.sidebar.warning("Sample data not found in data/adult.csv")

# --------------------------------------------------
# Upload Section (Styled Same as Sample Data)
# --------------------------------------------------
st.sidebar.subheader("📤 Upload Test CSV File")

uploaded_file = st.sidebar.file_uploader(
    "",
    type=["csv"]
)

st.sidebar.markdown(
    "<small>📌 CSV must contain all feature columns and the <b>income</b> target column.</small>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# Main Logic
# --------------------------------------------------
if uploaded_file is None:
    st.warning("⬅️ Upload a CSV file or download sample data to continue.")
else:
    data = pd.read_csv(uploaded_file)

    # Dataset Preview
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("📄 Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if "income" not in data.columns:
        st.error("❌ Uploaded CSV must contain the 'income' column.")
    else:
        model_path = f"models/{model_file_name}.pkl"
        model = joblib.load(model_path)

        X_test = data.drop("income", axis=1)
        y_test = data["income"].map({"<=50K": 0, ">50K": 1})

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # --------------------------------------------------
        # Performance Overview (INLINE MODEL NAME)
        # --------------------------------------------------
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader(f"📊 Performance Overview : {model_name_display}")

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
        # Classification Summary
        # --------------------------------------------------
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
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
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(model_name_display)

        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.success("✅ Evaluation completed successfully!")