import pandas as pd
from utils.preprocess import preprocess_text

print("Loading datasets...")
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

print("Adding labels...")
fake["label"] = 0
true["label"] = 1

print("Combining datasets...")
news = pd.concat([fake, true], ignore_index=True)

print("Shuffling...")
news = news.sample(frac=1, random_state=42).reset_index(drop=True)

print("Creating content column...")
news["content"] = news["title"] + " " + news["text"]
news = news[["content", "label"]]

print(f"Dataset size: {len(news)} articles")

print("Starting preprocessing...")
from tqdm import tqdm

tqdm.pandas()

news["content"] = news["content"].progress_apply(preprocess_text)
print("Preprocessing completed!")

print(news.isnull().sum())

news.dropna(inplace=True)

print(news["label"].value_counts())

news.to_csv("data/news.csv", index=False)

print("Processed dataset saved successfully!")