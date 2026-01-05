import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

nltk.download('stopwords')
nltk.download('wordnet')

# Load data
with open("gate_cs_pyqs.txt", "r", encoding="utf-8") as f:
    questions = f.readlines()

# Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

cleaned_questions = [clean_text(q) for q in questions]

# Vectorization
vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
X = vectorizer.fit_transform(cleaned_questions)

# Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# Save models (LOCAL ENV)
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(kmeans, "kmeans.pkl")

print("✅ Models saved successfully")
