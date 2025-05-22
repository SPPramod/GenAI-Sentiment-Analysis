import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analysis", layout="centered")

def add_background():
    st.markdown(
        """
        <style>
        html, body {
            height: 100%;
            background-image: url("https://images.unsplash.com/photo-1531746790731-6c087fecd65a");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
            color: black !important;
        }

        .stApp {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 10px;
            max-width: 800px;
            margin: auto;
            box-shadow: 0 0 20px rgba(0,0,0,0.2);
            color: black !important;
        }

        /* Force text color inside widgets */
        .stMarkdown, .stTextInput, .stTextArea, .stButton, .stAlert {
            color: black !important;
        }

        /* Override Streamlit's default grayish header colors */
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: black !important;
        }

        .stTextInput > div > input,
        .stTextArea > div > textarea,
        .stButton > button {
            background-color: white;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_background()

st.title("💬 Twitter Sentiment Analysis")
st.write("Analyze sentiment (Positive, Neutral, Negative) using a fine-tuned RoBERTa model.")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_pipeline = load_model()

user_input = st.text_area("Enter a tweet or short text:")

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text before submitting.")
    else:
        with st.spinner("Analyzing..."):
            results = sentiment_pipeline([user_input])
            sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            label_id = int(results[0]['label'].split('_')[-1])
            sentiment_label = sentiment_map[label_id]
            score = results[0]['score']

            st.success(f"**Sentiment:** {sentiment_label}")
            st.write(f"**Confidence:** {score:.2f}")