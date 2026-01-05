import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np

# Download stopwords (safe)
nltk.download('stopwords')

# Load trained models
vectorizer = joblib.load('vectorizer.pkl')
kmeans = joblib.load('kmeans.pkl')

stop_words = set(stopwords.words('english'))

# ----------------------------
# Text Cleaning (STABLE)
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="PYQ Insight Engine", layout="wide")

st.title("📘 PYQ Insight Engine")
st.write(
    "Analyze Computer Science PYQs and discover **dominant topics & trends** using NLP clustering."
)

input_text = st.text_area(
    "📥 Paste questions (one per line):",
    height=250
)

# ----------------------------
# Analysis Logic
# ----------------------------
if st.button("🔍 Analyze"):
    if input_text.strip() == "":
        st.warning("Please enter some questions.")
    else:
        questions = input_text.strip().split("\n")
        cleaned = [clean_text(q) for q in questions]

        X = vectorizer.transform(cleaned)
        clusters = kmeans.predict(X)

        df = pd.DataFrame({
            "Question": questions,
            "Detected Topic (Cluster)": clusters
        })

        # ----------------------------
        # Results Table
        # ----------------------------
        st.subheader("📊 Question-wise Topic Detection")
        st.dataframe(df, use_container_width=True)

        # ----------------------------
        # Distribution Plot
        # ----------------------------
        st.subheader("📈 Topic Distribution")
        fig, ax = plt.subplots(figsize=(4, 3))
        df["Detected Topic (Cluster)"].value_counts().sort_index().plot(
            kind="bar",
            ax=ax,
            color="#4F81BD"
        )

        ax.set_xlabel("Cluster")
        ax.set_ylabel("Questions")
        ax.set_title("PYQ Topic Distribution", fontsize=10)
        ax.tick_params(axis='x', rotation=0)
        plt.tight_layout(pad=1)

        st.pyplot(fig, use_container_width=False)

        # ----------------------------
        # 🔥 CLUSTER-WISE KEYWORDS (NEW FEATURE)
        # ----------------------------
        st.subheader("🧠 Cluster-wise Insights")

        feature_names = vectorizer.get_feature_names_out()
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

        for cluster_id in sorted(df["Detected Topic (Cluster)"].unique()):
            st.markdown(f"### 🔹 Cluster {cluster_id}")

            top_keywords = [
                feature_names[ind]
                for ind in order_centroids[cluster_id, :8]
            ]

            cluster_questions = df[
                df["Detected Topic (Cluster)"] == cluster_id
            ]["Question"].tolist()

            st.write("**Top Keywords:**", ", ".join(top_keywords))

            st.write(
                f"**Insight:** This cluster primarily focuses on concepts related to "
                f"**{', '.join(top_keywords[:3])}**, indicating a recurring theme in PYQs."
            )

            with st.expander("📌 Sample Questions"):
                for q in cluster_questions[:5]:
                    st.write("•", q)

