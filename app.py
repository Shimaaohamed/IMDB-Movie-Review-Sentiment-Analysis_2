# app.py
import streamlit as st
import joblib
from preprocessing import clean_text

st.set_page_config(page_title="IMDB Sentiment (Demo)", layout="centered")

st.title("🎬 IMDB Movie Review Sentiment")
st.write("ادخل مراجعة فيلم (review) وسيعرض التطبيق توقع المشاعر (positive/negative).")

# تحميل الموديل
@st.cache_resource
def load_model(path="model_imdb.pkl"):
    return joblib.load(path)

model = load_model()

text = st.text_area("Enter movie review here...", height=200)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text to predict.")
    else:
        clean = clean_text(text)
        pred = model.predict([clean])[0]
        # إذا أردت تحصل على احتمالات:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([clean])[0]
            if pred == "positive":
                score = proba[list(model.classes_).index("positive")]
            else:
                score = proba[list(model.classes_).index("negative")]
            st.write(f"**Prediction:** {pred}  —  **Confidence:** {score:.2f}")
        else:
            st.write(f"**Prediction:** {pred}")
