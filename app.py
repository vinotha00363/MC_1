import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Dynamic Earnings Manipulation Detection System (SVM)",
    layout="wide"
)

# --------------------------------------------------
# SIDEBAR – DATASET UPLOAD
# --------------------------------------------------
st.sidebar.title("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Earnings Manipulator Excel File",
    type=["xlsx"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This app trains an SVM model dynamically using the uploaded dataset."
)

# --------------------------------------------------
# MAIN TITLE
# --------------------------------------------------
st.markdown("""
# 📊 Dynamic Earnings Manipulation Detection System (SVM)

This application performs **end-to-end earnings manipulation analysis** using:

- Exploratory Data Analysis (EDA)
- Feature scaling
- **Support Vector Machine (SVM)**
- User-defined prediction
""")

st.info("⬅️ Please upload the dataset to proceed")

# --------------------------------------------------
# AFTER DATASET UPLOAD
# --------------------------------------------------
if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    st.success("Dataset uploaded successfully ✅")
    st.markdown("---")

    # --------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------
    st.subheader("🧠 Feature Selection")

    feature_cols = [
        "DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "ACCR", "LEVI"
    ]

    target_col = "Manipulator"

    if not all(col in df.columns for col in feature_cols + [target_col]):
        st.error("Dataset must contain required columns.")
        st.stop()

    X = df[feature_cols]
    y = df[target_col].map({"Yes": 1, "No": 0})

    # --------------------------------------------------
    # MODEL PIPELINE
    # --------------------------------------------------
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True))
    ])

    st.markdown("---")
    st.subheader("⚙️ Model Training")

    if st.button("▶ Train SVM Model"):

        with st.status("Training SVM model...", expanded=True):
            time.sleep(1)
            st.write("✔ Scaling features")
            time.sleep(1)
            st.write("✔ Training SVM classifier")

            svm_pipeline.fit(X, y)

            time.sleep(1)
            st.write("✔ Model training completed")

        st.success("SVM model trained successfully 🎉")

        # --------------------------------------------------
        # USER INPUT PREDICTION
        # --------------------------------------------------
        st.markdown("---")
        st.subheader("✍️ User-Defined Prediction")

        cols = st.columns(4)

        user_input = {}
        for i, col in enumerate(feature_cols):
            user_input[col] = cols[i % 4].number_input(col, value=1.0)

        input_df = pd.DataFrame([user_input])

        if st.button("🔍 Predict Manipulation Risk"):

            prob = svm_pipeline.predict_proba(input_df)[0][1]
            pred = svm_pipeline.predict(input_df)[0]

            st.markdown("---")
            st.subheader("📌 Final Result")

            if pred == 1:
                st.error(
                    f"⚠️ **EARNINGS MANIPULATOR DETECTED**\n\n"
                    f"Probability: **{prob:.2%}**"
                )
            else:
                st.success(
                    f"✅ **NON-MANIPULATOR**\n\n"
                    f"Probability: **{1 - prob:.2%}**"
                )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("© Earnings Manipulation Detection System | SVM Deployment")
