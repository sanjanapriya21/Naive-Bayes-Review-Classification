
import re
import streamlit as st
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Restaurant Review Sentiment Analysis",
    page_icon="🍽️",
    layout="centered"
)

st.title("🍽️ Restaurant Review Sentiment Analysis")

st.write(
    "Compare Bag of Words and TF-IDF using Bernoulli Naive Bayes "
    "and predict restaurant review sentiment."
)


@st.cache_resource
def train_and_compare_models():

    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    data = pd.read_csv("output.csv")
    data.columns = data.columns.str.strip()

    if not {"Review", "Liked"}.issubset(data.columns):
        raise ValueError("CSV must contain Review and Liked columns.")

    data = data.dropna(subset=["Review", "Liked"]).copy()

    data["Review"] = data["Review"].astype(str)
    data["Liked"] = pd.to_numeric(
        data["Liked"], errors="coerce"
    )

    data = data.dropna(subset=["Liked"]).copy()
    data["Liked"] = data["Liked"].astype(int)

    stop_words = set(stopwords.words("english"))

    # Preserve negative words
    stop_words.difference_update(
        {"no", "nor", "not", "never"}
    )

    lemmatizer = WordNetLemmatizer()

    def clean_review(review):

        review = str(review).lower()
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
    y = data["Liked"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    results = {}

    # MODEL 1: BAG OF WORDS + BERNOULLI NB

    bow_vectorizer = CountVectorizer(binary=True)

    X_train_bow = bow_vectorizer.fit_transform(X_train)
    X_test_bow = bow_vectorizer.transform(X_test)

    bow_model = BernoulliNB()

    bow_model.fit(X_train_bow, y_train)

    bow_predictions = bow_model.predict(X_test_bow)

    results["Bag of Words"] = {
        "vectorizer": bow_vectorizer,
        "model": bow_model,
        "accuracy": accuracy_score(y_test, bow_predictions),
        "precision": precision_score(
            y_test, bow_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, bow_predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, bow_predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, bow_predictions
        ),
        "report": classification_report(
            y_test, bow_predictions, zero_division=0
        )
    }

    # MODEL 2: TF-IDF + BERNOULLI NB

    tfidf_vectorizer = TfidfVectorizer()

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    tfidf_model = BernoulliNB()

    tfidf_model.fit(X_train_tfidf, y_train)

    tfidf_predictions = tfidf_model.predict(X_test_tfidf)

    results["TF-IDF"] = {
        "vectorizer": tfidf_vectorizer,
        "model": tfidf_model,
        "accuracy": accuracy_score(y_test, tfidf_predictions),
        "precision": precision_score(
            y_test, tfidf_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, tfidf_predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, tfidf_predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, tfidf_predictions
        ),
        "report": classification_report(
            y_test, tfidf_predictions, zero_division=0
        )
    }

    # SELECT BEST MODEL BASED ON ACCURACY

    best_name = max(
        results,
        key=lambda name: results[name]["accuracy"]
    )

    return results, best_name, clean_review, len(data)


# TRAIN MODELS

try:

    with st.spinner("Training and comparing models..."):

        results, best_name, clean_review, row_count = (
            train_and_compare_models()
        )

except Exception as error:

    st.error("Model training failed. Please check the dataset and logs.")
    st.exception(error)
    st.stop()


# MODEL COMPARISON

st.subheader("Model Performance Comparison")

comparison = pd.DataFrame(
    {
        "Accuracy": [
            results["Bag of Words"]["accuracy"],
            results["TF-IDF"]["accuracy"]
        ],
        "Precision": [
            results["Bag of Words"]["precision"],
            results["TF-IDF"]["precision"]
        ],
        "Recall": [
            results["Bag of Words"]["recall"],
            results["TF-IDF"]["recall"]
        ],
        "F1 Score": [
            results["Bag of Words"]["f1"],
            results["TF-IDF"]["f1"]
        ]
    },
    index=["Bag of Words", "TF-IDF"]
)

st.dataframe(
    comparison.style.format("{:.2%}"),
    use_container_width=True
)


# DISPLAY BEST MODEL

st.success(f"Best Performing Model: {best_name}")

st.write(
    f"Dataset records used: {row_count}"
)

st.write(
    f"Selected model accuracy: "
    f"{results[best_name]['accuracy']:.2%}"
)

best_result = results[best_name]


# MODEL EVALUATION

with st.expander("View Confusion Matrix and Classification Reports"):

    for name in ["Bag of Words", "TF-IDF"]:

        st.subheader(name + " - Bernoulli Naive Bayes")

        st.write(
            "Accuracy:",
            f"{results[name]['accuracy']:.2%}"
        )

        st.write("Confusion Matrix")

        st.dataframe(
            pd.DataFrame(
                results[name]["confusion_matrix"],
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"]
            )
        )

        st.write("Classification Report")

        st.code(results[name]["report"])


# PREDICT NEW REVIEWS

st.subheader("Predict Sentiment of New Reviews")

review = st.text_area(
    "Enter your restaurant review",
    placeholder="The food was fantastic and service was excellent!",
    height=120
)

if st.button(
    "Predict Sentiment",
    type="primary",
    use_container_width=True
):

    if not review.strip():

        st.warning("Please enter a review.")

    else:

        cleaned_review = clean_review(review)

        review_features = best_result["vectorizer"].transform(
            [cleaned_review]
        )

        prediction = int(
            best_result["model"].predict(review_features)[0]
        )

        st.subheader("Prediction Result")

        if prediction == 1:

            st.success("😊 Positive Review")

        else:

            st.error("☹️ Negative Review")

        st.write("Selected Model:", best_name)

        st.write(
            "Model Accuracy:",
            f"{best_result['accuracy']:.2%}"
        )


# PROJECT INFORMATION

with st.expander("About This Project"):

    st.write(
        "This project performs restaurant review sentiment "
        "classification using Natural Language Processing."
    )

    st.write(
        "Bag of Words and TF-IDF are compared using "
        "Bernoulli Naive Bayes."
    )

    st.write(
        "The best representation is selected based on "
        "test accuracy and used for predicting new reviews."
    )
