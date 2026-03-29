import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import re
from stop_words import get_stop_words
import emoji

DATASET_PATH = "dataset/toxic_comments.csv"
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "logistic_model.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=["text", "label"])

russian_stopwords = set(get_stop_words("ru"))

def preprocess(text):
    text = text.lower()
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@[\w_]+", "", text)
    text = re.sub(r"#[\w_]+", "", text)
    text = re.sub(r"[^а-яa-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [word for word in words if word not in russian_stopwords]
    return " ".join(words)

df["text_clean"] = df["text"].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(
    df["text_clean"], df["label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight='balanced', C=2.0)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, MODEL_FILE)
joblib.dump(vectorizer, VECTORIZER_FILE)
print(f"Модель сохранена в {MODEL_FILE}")
print(f"Векторизатор сохранён в {VECTORIZER_FILE}")
