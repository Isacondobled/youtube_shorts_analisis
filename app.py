import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="YouTube Shorts Success Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 YouTube Shorts Success Predictor")
st.markdown(
    """
    This app predicts whether a **YouTube Short will be successful** based on
    **pre-upload features only and data available**.
    """
)

@st.cache_resource
def load_model():
    return joblib.load("models/model_gb.pkl")

model = load_model()

THRESHOLD = 0.5

st.sidebar.header("🎛️ Short Configuration")

duration_sec = st.sidebar.slider("Duration (seconds)", 5, 60, 20)
hashtags_count = st.sidebar.slider("Number of hashtags", 0, 10, 3)
upload_hour = st.sidebar.slider("Upload hour", 0, 23, 18)
category = st.sidebar.selectbox(
    "Category",
    options=["Education", "Entertainment", "Gaming", "Lifestyle", "Music", "Tech"]
)

category_mapping = {
    "Education": 0,
    "Entertainment": 1,
    "Gaming": 2,
    "Lifestyle": 3,
    "Music": 4,
    "Tech": 5
}

input_data = pd.DataFrame({
    "duration_sec": [duration_sec],
    "hashtags_count": [hashtags_count],
    "upload_hour": [upload_hour],
    "category": [category_mapping[category]]
})

st.subheader("📥 Input Features")
st.dataframe(input_data, use_container_width=True)

if st.button("🚀 Predict Success"):
    prob = model.predict_proba(input_data)[:, 1][0]
    prediction = int(prob >= THRESHOLD)

    st.subheader("📈 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Success Probability", f"{prob:.2%}")

    with col2:
        st.metric("Prediction", "✅ Successful" if prediction else "❌ Not Successful")

    st.markdown("---")
    st.markdown(f"**Decision threshold:** `{THRESHOLD}`")

st.markdown(
    """
    ---
    ### 📌 Notes
    - Engagement metrics (likes, views, comments, shares) are **not used**
      because they are unavailable before publishing.
    """
)