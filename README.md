# 📘 PYQ Insight Engine

An NLP-powered tool to analyze and cluster Computer Science previous year questions (PYQs) using unsupervised learning.

## 🚀 Features
- Cleans and preprocesses PYQs using NLP
- Clusters questions using TF-IDF + KMeans
- Interactive Streamlit UI
- Cluster-wise topic distribution visualization
- Helps identify dominant trends in exam questions

## 🛠 Tech Stack
- Python
- Scikit-learn
- NLP (NLTK)
- Streamlit
- Matplotlib

## 📊 How it Works
1. Paste multiple PYQs (one per line)
2. Model predicts topic cluster
3. Displays topic-wise distribution
4. Shows key insights visually

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
