import streamlit as st
import joblib

from utils.preprocess import preprocess_text
from ir.news_api import get_related_news
from ir.retrieve import rank_articles
from summarizer.summarize import summarize_text

# ---------------------------
# Load trained model
# ---------------------------
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Fake News Detector & News Summarizer")
st.write("Paste a news article or headline to analyze.")

user_input = st.text_area(
    "News Article",
    height=250,
    placeholder="Paste the news article here..."
)

# ---------------------------
# Analyze Button
# ---------------------------
if st.button("Analyze"):

    if not user_input.strip():
        st.warning("Please enter a news article.")
        st.stop()

    with st.spinner("Analyzing article..."):

        # ---------------------------
        # Fake News Prediction
        # ---------------------------
        clean_text = preprocess_text(user_input)

        vector = vectorizer.transform([clean_text])

        prediction = model.predict(vector)[0]

        st.divider()

        if prediction == 1:
            st.success("✅ Prediction: REAL NEWS")
        else:
            st.error("❌ Prediction: FAKE NEWS")

        # ---------------------------
        # Information Retrieval
        # ---------------------------
        articles = get_related_news(user_input, page_size=10)

        ranked_articles = rank_articles(
            user_input,
            articles
        )

        # ---------------------------
        # Credibility Score
        # ---------------------------
        if ranked_articles:

            credibility = ranked_articles[0]["similarity"] * 100

            credibility = min(100, round(credibility, 2))

        else:

            credibility = 0

        st.metric(
            "Credibility Score",
            f"{credibility}%"
        )

        # ---------------------------
        # Summary
        # ---------------------------
        st.subheader("Summary")

        summary = summarize_text(
            user_input,
            sentences_count=3
        )

        st.write(summary)

        # ---------------------------
        # Supporting Articles
        # ---------------------------
        st.subheader("Related Articles")

        if ranked_articles:

            for article in ranked_articles[:5]:

                st.markdown(f"### {article['title']}")

                st.write(
                    f"Similarity: {round(article['similarity']*100,2)}%"
                )

                st.markdown(
                    f"[Read Full Article]({article['url']})"
                )

                st.divider()

        else:

            st.info("No related articles found.")