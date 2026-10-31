
    # TF-IDF + Bernoulli Naive Bayes
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
        "precision": precision_score(y_test, tfidf_predictions, zero_division=0),
        "recall": recall_score(y_test, tfidf_predictions, zero_division=0),
        "f1": f1_score(y_test, tfidf_predictions, zero_division=0),
        "predictions": tfidf_predictions,
        "confusion_matrix": confusion_matrix(y_test, tfidf_predictions),
        "report": classification_report(y_test, tfidf_predictions, zero_division=0),
    }

    # Choose the representation with the higher test accuracy.
    best_name = max(results, key=lambda name: results[name]["accuracy"])

    return results, best_name, clean_review, len(data)


try:
    with st.spinner("Training and comparing both Bernoulli Naïve Bayes models..."):
        results, best_name, clean_review, row_count = train_and_compare_models()
except Exception as error:
    st.error("The model could not be trained. Please check output.csv and the app logs.")
    st.exception(error)
    st.stop()

st.subheader("Model comparison")
comparison = pd.DataFrame(
    {
        "Representation": ["Bag of Words", "TF-IDF"],
        "Accuracy": [results["Bag of Words"]["accuracy"], results["TF-IDF"]["accuracy"]],
        "Precision": [results["Bag of Words"]["precision"], results["TF-IDF"]["precision"]],
        "Recall": [results["Bag of Words"]["recall"], results["TF-IDF"]["recall"]],
        "F1-score": [results["Bag of Words"]["f1"], results["TF-IDF"]["f1"]],
    }
).set_index("Representation")

st.dataframe(
    comparison.style.format("{:.2%}"),
    use_container_width=True
)

st.success(f"Best representation: {best_name}")
st.caption(
    f"Selected using test accuracy. Both models use Bernoulli Naïve Bayes. "
    f"Dataset rows used: {row_count}."
)

best_result = results[best_name]

with st.expander("View evaluation details for both models"):
    for name in ["Bag of Words", "TF-IDF"]:
        st.markdown(f"**{name} + Bernoulli Naïve Bayes**")
        st.write(f"Accuracy: {results[name]['accuracy']:.2%}")
        st.write("Confusion matrix:")
        st.dataframe(
            pd.DataFrame(
                results[name]["confusion_matrix"],
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"],
            )
        )
        st.text("Classification report:")
        st.code(results[name]["report"])

st.subheader("Predict sentiment of a new review")
review = st.text_area(
    "Enter your restaurant review",
    placeholder="Example: The food was fantastic and the service was excellent!",
    height=120,
)

if st.button("Predict Sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        cleaned_review = clean_review(review)
        review_features = best_result["vectorizer"].transform([cleaned_review])
        prediction = int(best_result["model"].predict(review_features)[0])

        st.subheader("Prediction")
        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("☹️ Negative Review")

        st.caption(f"Used: {best_name} + Bernoulli Naïve Bayes")
        st.caption(f"Test accuracy of selected model: {best_result['accuracy']:.2%}")

with st.expander("About this project"):
    st.write(
        "The application preprocesses restaurant reviews, compares Bag of Words "
        "and TF-IDF representations using Bernoulli Naïve Bayes, evaluates both "
        "on the same 80:20 train-test split, and selects the representation "
        "with the higher test accuracy for new-review prediction."
    )

