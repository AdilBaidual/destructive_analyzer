import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Отключаем GUI
import matplotlib.pyplot as plt
import joblib
import os
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")


# === Загрузка модели и векторизатора ===
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

    # === Распределение по классам ===
    plt.figure(figsize=(6, 4))
    plt.bar(["Нейтральный", "Деструктивный"], [neutral, destructive], color=["green", "red"])
    plt.title("Распределение предсказанных классов")
    plt.ylabel("Количество сообщений")
    plot_path = "static/class_distribution.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # === Гистограмма вероятностей ===
    plt.figure(figsize=(8, 4))
    plt.hist(df["probability"], bins=20, color='purple', edgecolor='black')
    plt.title("Распределение вероятностей деструктивности")
    plt.xlabel("Вероятность")
    plt.ylabel("Количество сообщений")
    plt.tight_layout()
    prob_dist_path = "static/prob_distribution.png"
    plt.savefig(prob_dist_path)
    plt.close()

    # === Словооблако ===
    all_text = " ".join(df["text_clean"].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    wordcloud_path = "static/wordcloud.png"
    plt.savefig(wordcloud_path)
    plt.close()

    # === Частотный анализ (топ-10 слов) ===
    russian_stopwords = stopwords.words("russian")
    count_vectorizer = CountVectorizer(stop_words=russian_stopwords, max_features=1000)
    word_counts = count_vectorizer.fit_transform(df["text_clean"].fillna(""))
    word_sum = word_counts.sum(axis=0).A1
    words = count_vectorizer.get_feature_names_out()
    freq_df = pd.DataFrame({"word": words, "count": word_sum})
    top_words = freq_df.sort_values(by="count", ascending=False).head(10)

    plt.figure(figsize=(8, 4))
    plt.barh(top_words["word"], top_words["count"], color="steelblue")
    plt.xlabel("Частота")
    plt.title("Топ-10 наиболее частотных слов")
    plt.gca().invert_yaxis()
    freq_path = "static/frequency_plot.png"
    plt.tight_layout()
    plt.savefig(freq_path)
    plt.close()

    # === Оценка по датам ===
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors='coerce')
        timeline = df.groupby(df["created_at"].dt.floor("h"))["probability"].mean().reset_index()

        plt.figure(figsize=(10, 4))
        plt.plot(timeline["created_at"], timeline["probability"], marker='o', linestyle='-')
        plt.title("Оценка вероятности деструктивности по времени")
        plt.xlabel("Время")
        plt.ylabel("Средняя вероятность")
        plt.xticks(rotation=45)
        plt.tight_layout()
        timeline_path = "static/timeline_plot.png"
        plt.savefig(timeline_path)
        plt.close()
    else:
        timeline_path = ""

    # === Возвращаем сводку ===
    return {
        "total": total,
        "neutral": neutral,
        "destructive": destructive,
        "avg_prob": avg_prob,
        "plot_path": plot_path,
        "wordcloud_path": wordcloud_path,
        "freq_path": freq_path,
        "timeline_path": timeline_path,
        "prob_dist_path": prob_dist_path
    }
