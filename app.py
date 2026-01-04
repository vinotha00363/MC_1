import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Dynamic Earnings Manipulation Detection (SVM)",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("📊 Dynamic Earnings Manipulation Detection System (SVM)")
st.write(
    "This application detects earnings manipulation using a "
    "**Support Vector Machine (SVM)** model trained on uploaded financial data."
)

st.markdown("---")

# --------------------------------------------------
# SIDEBAR – DATASET UPLOAD
# --------------------------------------------------
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Earnings Manipulator Excel File (.xlsx)",
    type=["xlsx"]
)

# --------------------------------------------------
# REQUIRED COLUMNS
# --------------------------------------------------
FEATURE_COLS = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "ACCR", "LEVI"]
TARGET_COL = "Manipulator"

# --------------------------------------------------
# AFTER DATASET UPLOAD
# --------------------------------------------------
if uploaded_file is not None:

    # Load data
    df = pd.read_excel(uploaded_file)
    st.success("Dataset uploaded successfully ✅")

    # Validate columns
    if not all(col in df.columns for col in FEATURE_COLS + [TARGET_COL]):
        st.error(
            "Dataset must contain the following columns:\n\n"
            f"{FEATURE_COLS + [TARGET_COL]}"
        )
        st.stop()

    # Preview
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.markdown("---")

    # Prepare data
    X = df[FEATURE_COLS]
    y = df[TARGET_COL].map({"Yes": 1, "No": 0})

    # --------------------------------------------------
    # TRAIN SVM MODEL
    # --------------------------------------------------
    st.subheader("⚙️ Train SVM Model")

    if st.button("▶ Train SVM Model"):

        svm_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True))
        ])

        svm_pipeline.fit(X, y)

        # Store model in session state
        st.session_state["svm_model"] = svm_pipeline
        st.session_state["svm_trained"] = True

        st.success("SVM model trained successfully 🎉")

    st.markdown("---")

    # --------------------------------------------------
    # USER-DEFINED PREDICTION
    # --------------------------------------------------
    st.subheader("✍️ User-Defined Prediction")

    cols = st.columns(4)
    user_input = {}

    for i, col in enumerate(FEATURE_COLS):
        user_input[col] = cols[i % 4].number_input(col, value=1.0)

    input_df = pd.DataFrame([user_input])

    # --------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------
    if st.button("🔍 Predict Manipulation Risk"):

        if "svm_trained" not in st.session_state:
            st.warning("⚠️ Please train the SVM model first.")
            st.stop()

        model = st.session_state["svm_model"]

        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        st.markdown("---")
        st.subheader("📌 Final Classification Result")

        if pred == 1:
            st.error(
                f"⚠️ **EARNINGS MANIPULATOR**\n\n"
                f"Predicted Probability: **{prob:.2%}**"
            )
        else:
            st.success(
                f"✅ **NON-MANIPULATOR**\n\n"
                f"Predicted Probability: **{1 - prob:.2%}**"
            )

        st.info("Model Used: **Support Vector Machine (SVM)**")

else:
    st.info("⬅️ Please upload the dataset to proceed")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("© Earnings Manipulation Detection System | SVM Deployment")
