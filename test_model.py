import joblib

from utils.preprocess import preprocess_text

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

text = input("Enter news: ")

clean = preprocess_text(text)

vector = vectorizer.transform([clean])

prediction = model.predict(vector)[0]

print("Prediction:", prediction)