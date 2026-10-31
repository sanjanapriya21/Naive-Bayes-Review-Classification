import re
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Restaurant Review Sentiment", page_icon="🍽️", layout="centered")

@st.cache_resource
def prepare_model():
    # Download the language resources needed for the same preprocessing used in the notebook.
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    data = pd.read_csv("output.csv")
    data.columns = data.columns.str.strip()

    if not {"Review", "Liked"}.issubset(data.columns):
        raise ValueError("The CSV must contain Review and Liked columns.")

    data = data.dropna(subset=["Review", "Liked"]).copy()
    data["Review"] = data["Review"].astype(str)

    stop_words = set(stopwords.words("english"))
    stop_words.difference_update({"no", "nor", "not", "never"}))
    lemmatizer = WordNetLemmatizer()

    def clean_review(review):
        review = review.lower()
        review = re.sub(r"[^a-zA-Z\s]", " ", review)
        review = re.sub(r"\d+", " ", review)
        review = re.sub(r"\s+", " ", review).strip()
        words = [
            lemmatizer.lemmatize(word)
            for word in review.split()
            if word not in stop_words
        ]
        return " ".join(words)

    data["clean_review"] = data["Review"].apply(clean_review)
    X = data["clean_review"]
    y = data["Liked"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    bow_vectorizer = CountVectorizer()
    X_train_bow = bow_vectorizer.fit_transform(X_train)
    X_test_bow = bow_vectorizer.transform(X_test)
    bow_model = BernoulliNB()
    bow_model.fit(X_train_bow, y_train)
    bow_accuracy = accuracy_score(y_test, bow_model.predict(X_test_bow))

    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    tfidf_model = BernoulliNB()
    tfidf_model.fit(X_train_tfidf, y_train)
    tfidf_accuracy = accuracy_score(y_test, tfidf_model.predict(X_test_tfidf))

    return {
        "bow": (bow_vectorizer, bow_model, bow_accuracy),
        "tfidf": (tfidf_vectorizer, tfidf_model, tfidf_accuracy),
        "clean_review": clean_review,
        "rows": len(data),
    }

st.title("🍽️ Restaurant Review Sentiment Analysis")
st.write("Enter a restaurant review and let the Naïve Bayes model predict its sentiment.")

try:
    models = prepare_model()
except Exception as error:
    st.error("The model could not be prepared. Check that output.csv is uploaded with app.py.")
    st.exception(error)
    st.stop()

model_choice = st.selectbox(
    "Choose a model representation",
    ["TF-IDF", "Bag of Words"]
)

review = st.text_area(
    "Enter your restaurant review",
    placeholder="Example: The food was fantastic and the service was excellent!",
    height=130
)

if st.button("Predict Sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        key = "tfidf" if model_choice == "TF-IDF" else "bow"
        vectorizer, model, accuracy = models[key]
        cleaned = models["clean_review"](review)
        features = vectorizer.transform([cleaned])
        prediction = int(model.predict(features)[0])

        st.subheader("Prediction")
        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("☹️ Negative Review")

        st.caption(f"Model: Bernoulli Naïve Bayes with {model_choice}")
        st.caption(f"Test accuracy: {accuracy:.2%}")

with st.expander("About this project"):
    st.write(
        "This application uses the restaurant review dataset from the project. "
        "It applies lowercase conversion, punctuation and number removal, "
        "stopword removal and lemmatization, followed by CountVectorizer or "
        "TF-IDF and a Bernoulli Naïve Bayes classifier."
    )
    st.write(f"Dataset records used: {models['rows']}")
