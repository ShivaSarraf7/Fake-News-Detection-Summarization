import joblib

model = joblib.load("models/fake_news_model.pkl")

print(model)