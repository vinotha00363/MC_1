import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Earnings Manipulation Detection (SVM)",
    layout="wide"
)

# --------------------------------------------------
# INIT SESSION STATE (CRITICAL FIX)
# --------------------------------------------------
if "svm_model" not in st.session_state:
    st.session_state.svm_model = None

if "svm_trained" not in st.session_state:
    st.session_state.svm_trained = False

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("📊 Earnings Manipulation Detection System (SVM)")
st.write(
    "This system detects earnings manipulation using a "
    "**Support Vector Machine (SVM)** model."
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

FEATURE_COLS = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "ACCR", "LEVI"]
TARGET_COL = "Manipulator"

# --------------------------------------------------
# DATASET PROCESSING
# --------------------------------------------------
if uploaded_file is None:
    st.info("⬅️ Please upload the dataset to proceed")
    st.stop()

df = pd.read_excel(uploaded_file)

# Validate columns
required_cols = FEATURE_COLS + [TARGET_COL]
if not all(col in df.columns for col in required_cols):
    st.error(f"Dataset must contain columns: {required_cols}")
    st.stop()

st.success("Dataset uploaded successfully ✅")

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

X = df[FEATURE_COLS]
y = df[TARGET_COL].map({"Yes": 1, "No": 0})

st.markdown("---")

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
st.subheader("⚙️ Model Training")

if st.button("▶ Train SVM Model"):

    with st.status("Training SVM model...", expanded=True) as status:
        st.write("✔ Scaling features")
        st.write("✔ Training SVM classifier")

        svm_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True))
        ])

        svm_pipeline.fit(X, y)

        st.session_state.svm_model = svm_pipeline
        st.session_state.svm_trained = True
        st.session_state.prediction_done = False

        status.update(label="Model training completed", state="complete")

    st.success("SVM model trained successfully 🎉")

st.markdown("---")

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
st.subheader("✍️ User-Defined Prediction")

cols = st.columns(4)
user_input = {}

for i, col in enumerate(FEATURE_COLS):
    user_input[col] = cols[i % 4].number_input(col, value=1.0)

input_df = pd.DataFrame([user_input])

# --------------------------------------------------
# PREDICTION (FINAL FIX)
# --------------------------------------------------
if st.button("🔍 Predict Manipulation Risk"):

    if not st.session_state.svm_trained:
        st.warning("⚠️ Please train the SVM model first.")
        st.stop()

    model = st.session_state.svm_model
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.session_state.prediction_done = True
    st.session_state.pred_value = pred
    st.session_state.pred_prob = prob

# --------------------------------------------------
# SHOW RESULT (PERSISTS AFTER RERUN)
# --------------------------------------------------
if st.session_state.prediction_done:

    st.markdown("---")
    st.subheader("📌 Final Classification Result")

    if st.session_state.pred_value == 1:
        st.error(
            f"⚠️ **EARNINGS MANIPULATOR**\n\n"
            f"Predicted Probability: **{st.session_state.pred_prob:.2%}**"
        )
    else:
        st.success(
            f"✅ **NON-MANIPULATOR**\n\n"
            f"Predicted Probability: **{1 - st.session_state.pred_prob:.2%}**"
        )

    st.info("Model Used: **Support Vector Machine (SVM)**")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("© Earnings Manipulation Detection System | SVM Deployment")
