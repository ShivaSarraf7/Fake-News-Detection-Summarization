import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report

import joblib


# Load processed dataset
print("Loading dataset...")

news = pd.read_csv("data/news.csv")


# Remove empty rows
news.dropna(inplace=True)


X = news["content"]
y = news["label"]


# Train-test split
print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# TF-IDF conversion
print("Creating TF-IDF vectors...")

tfidf = TfidfVectorizer(
    max_features=5000
)


X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)


# Models
models = {

    "Logistic Regression":
    LogisticRegression(max_iter=1000),

    "Naive Bayes":
    MultinomialNB(),

    "Linear SVM":
    LinearSVC(),

    "Random Forest":
    RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


results = {}


# Training
for name, model in models.items():

    print("\nTraining:", name)

    model.fit(
        X_train_tfidf,
        y_train
    )

    prediction = model.predict(
        X_test_tfidf
    )

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    results[name] = accuracy


    print("Accuracy:", accuracy)

    print(
        classification_report(
            y_test,
            prediction
        )
    )


print("\nModel Comparison")

for model, score in results.items():
    print(model, ":", score)


# Select best model
best_model_name = max(
    results,
    key=results.get
)


print(
    "\nBest Model:",
    best_model_name
)


best_model = models[best_model_name]


# Save model and vectorizer

joblib.dump(
    best_model,
    "models/fake_news_model.pkl"
)

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)


print("Model saved successfully!")