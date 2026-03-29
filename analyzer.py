import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")

model = joblib.load("model/logistic_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

def analyze_file(filename):
    filepath = os.path.join("prepared_data", filename)
    df = pd.read_csv(filepath)

    if "text_clean" not in df.columns:
        raise ValueError("Отсутствует колонка 'text_clean' в подготовленных данных.")

    X = vectorizer.transform(df["text_clean"].astype(str))
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    df["pred"] = preds
    df["probability"] = probs

    total = len(df)
    destructive = int((df["pred"] == 1).sum())
    neutral = total - destructive
    avg_prob = probs.mean()

    plt.figure(figsize=(6, 4))
    plt.bar(["Нейтральный", "Деструктивный"], [neutral, destructive], color=["green", "red"])
    plt.title("Распределение предсказанных классов")
    plt.ylabel("Количество сообщений")
    plot_path = "static/class_distribution.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(df["probability"], bins=20, color='purple', edgecolor='black')
    plt.title("Распределение вероятностей деструктивности")
    plt.xlabel("Вероятность")
    plt.ylabel("Количество сообщений")
    plt.tight_layout()
    prob_dist_path = "static/prob_distribution.png"
    plt.savefig(prob_dist_path)
    plt.close()

    russian_stopwords = stopwords.words("russian")
    count_vectorizer = CountVectorizer(stop_words=russian_stopwords, max_features=1000)

    destructive_texts = df[df["pred"] == 1]["text_clean"].fillna("")
    word_counts = count_vectorizer.fit_transform(destructive_texts)
    word_sum = word_counts.sum(axis=0).A1
    words = count_vectorizer.get_feature_names_out()
    freq_df = pd.DataFrame({"word": words, "count": word_sum})
    top_words = freq_df.sort_values(by="count", ascending=False).head(10)

    plt.figure(figsize=(8, 4))
    plt.barh(top_words["word"], top_words["count"], color="steelblue")
    plt.xlabel("Частота")
    plt.title("Топ-10 наиболее частотных слов (деструктивный контент)")
    plt.gca().invert_yaxis()
    freq_path = "static/frequency_plot.png"
    plt.tight_layout()
    plt.savefig(freq_path)
    plt.close()

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors='coerce')
        df["date"] = df["created_at"].dt.date
        timeline = df.groupby("date")["probability"].mean().reset_index()

        plt.figure(figsize=(10, 4))
        plt.plot(timeline["date"], timeline["probability"], marker='o', linestyle='-')
        plt.title("Оценка вероятности деструктивности по датам")
        plt.xlabel("Дата")
        plt.ylabel("Средняя вероятность")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        timeline_path = "static/timeline_plot.png"
        plt.savefig(timeline_path)
        plt.close()
    else:
        timeline_path = ""

    return {
        "total": total,
        "neutral": neutral,
        "destructive": destructive,
        "avg_prob": avg_prob,
        "plot_path": plot_path,
        "freq_path": freq_path,
        "timeline_path": timeline_path,
        "prob_dist_path": prob_dist_path
    }
