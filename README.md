# Destructive Analyzer

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python)](https://python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web_Framework-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-DL-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org/)

ML-система для анализа деструктивного контента в Telegram-каналах. Две модели: для определения деструктивности и экстремизма. Есть веб-интерфейс и возможность дообучать модели.

## Возможности

- Парсинг сообщений из Telegram-каналов
- Классификация на деструктив/не деструктив
- Детекция экстремистского контента
- Визуализация: графики, wordcloud
- Дообучение моделей на новых данных
- Веб-интерфейс на Flask

## Структура

```
destructive_analyzer/
├── app.py                   # Flask приложение
├── analyzer.py              # Логика анализа
├── preprocessor.py          # Очистка текста
├── train_model.py           # Обучение моделей
├── tg_parser.py             # Telegram парсер
├── config.py                # Конфигурация
├── model/                   # Сохранённые модели
├── dataset/                 # Датасеты
├── raw_data/                # Сырые данные
├── prepared_data/           # Обработанные данные
├── static/                  # CSS/JS
└── templates/               # HTML шаблоны
```

## Как это работает

**Две модели:**
1. **SGDClassifier** — классифицирует деструктивность (TF-IDF)
2. **Keras Sequential** — нейросеть для экстремизма (Embedding слои)

**Обработка текста:**
- Удаляем ссылки, упоминания, спецсимволы
- Лемматизация через pymorphy2
- TF-IDF векторизация (5000 признаков)

## Запуск

```bash
# Создать окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt

# Настроить Telegram API
cp .env.example .env
# Отредактировать .env (API_ID и API_HASH с my.telegram.org)

# Авторизоваться в Telegram
python tg_parser.py

# Обучить модели
python train_model.py

# Запустить
python app.py
```

Открыть в браузере: `http://localhost:5000`

## ML Модели

**Деструктивность:**
- Алгоритм: SGDClassifier
- Векторизация: TF-IDF (5000 фичей)
- Precision: 0.87, Recall: 0.82

**Экстремизм:**
- Архитектура: Embedding → GlobalMaxPooling → Dense
- Токенизатор: Keras (10000 слов)
- Accuracy: 0.91

## API

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/` | Главная |
| POST | `/analyze` | Анализ канала |
| POST | `/train` | Дообучение |
| GET | `/results/<id>` | Результаты |

## Зависимости

```
flask>=2.0.0
scikit-learn>=1.0.0
tensorflow>=2.10.0
telethon>=1.25.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
wordcloud>=1.8.0
pymorphy2>=0.9.0
```

## Важно

- Не коммить `.env` файлы
- Храни `session.session` в безопасности
- Используй отдельный Telegram аккаунт для парсинга
