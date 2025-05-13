import pandas as pd
import os
import re
import emoji
from stop_words import get_stop_words

# === Настройки ===
RAW_DATA_FOLDER = "raw_data"
OUTPUT_FOLDER = "prepared_data"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Предобработка текста ===
russian_stopwords = set(get_stop_words("ru"))

def preprocess(text):
    text = str(text).lower()
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@[\w_]+", "", text)
    text = re.sub(r"#[\w_]+", "", text)
    text = re.sub(r"[^а-яa-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [word for word in words if word not in russian_stopwords]
    return " ".join(words)

def process_and_save(file_name):
    input_path = os.path.join(RAW_DATA_FOLDER, file_name)
    name_without_ext = os.path.splitext(file_name)[0]
    output_file = os.path.join(OUTPUT_FOLDER, f"{name_without_ext}_prepared.csv")

    df = pd.read_csv(input_path)

    if "text_clean" not in df.columns:
        df["text_clean"] = df["text"].apply(preprocess)
        df = df[df["text_clean"].str.strip().astype(bool)]
        df.to_csv(output_file, index=False)
        print(f"Обработанные данные сохранены в {output_file}")
    else:
        print("Файл уже содержит колонку 'text_clean'. Обработка не требуется.")
        output_file = input_path  # не создаём новый файл

    return os.path.basename(output_file)

# === Пример запуска ===
if __name__ == "__main__":
    input_filename = "20240513_channelname.csv"  # пример
    result_file = process_and_save(input_filename)
    print("Результирующий файл:", result_file)
