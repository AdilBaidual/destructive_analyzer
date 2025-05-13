from flask import Flask, request, render_template_string
import os
import matplotlib
matplotlib.use('Agg')

from preprocessor import process_and_save
from analyzer import analyze_file
from tg_parser import parse_telegram_channel, parse_single_post

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Анализ Telegram-контента</title>
</head>
<body>
    <h2>Выберите режим анализа</h2>
    <form method="post">
        <label><input type="radio" name="mode" value="channel" checked> Последние N постов</label><br>
        Название канала: <input type="text" name="channel"><br>
        Кол-во постов: <input type="number" name="count" min="1" max="100"><br><br>

        <label><input type="radio" name="mode" value="post"> Один пост по ID</label><br>
        Название канала: <input type="text" name="single_channel"><br>
        ID поста: <input type="number" name="post_id" min="1"><br><br>

        <input type="submit" value="Анализировать">
    </form>

    {% if summary %}
        <h3>Результаты анализа</h3>
        <p>Всего сообщений: {{ summary.total }}</p>
        <p>Нейтральных сообщений: {{ summary.neutral }}</p>
        <p>Деструктивных сообщений: {{ summary.destructive }}</p>
        <p>Средняя вероятность деструктивности: {{ summary.avg_prob | default("н/д") }}</p>
        <h4>Графики:</h4>
        <img src="/{{ summary.plot_path }}" alt="График распределения"><br>
        <img src="/{{ summary.prob_dist_path }}" alt="Распределение вероятностей"><br>
        <img src="/{{ summary.wordcloud_path }}" alt="Словооблако"><br>
        <img src="/{{ summary.freq_path }}" alt="Частотный анализ"><br>
        <img src="/{{ summary.timeline_path }}" alt="Оценка по датам"><br>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None

    if request.method == "POST":
        mode = request.form.get("mode")

        if mode == "channel":
            channel = request.form["channel"].strip()
            count = int(request.form["count"])
            raw_filename = parse_telegram_channel(channel, count)
        elif mode == "post":
            channel = request.form["single_channel"].strip()
            post_id = int(request.form["post_id"])
            raw_filename = parse_single_post(channel, post_id)
        else:
            return "Неверный режим", 400

        prepared_file = process_and_save(raw_filename)
        summary = analyze_file(prepared_file)

    return render_template_string(HTML_TEMPLATE, summary=summary)


if __name__ == "__main__":
    os.makedirs("raw_data", exist_ok=True)
    os.makedirs("prepared_data", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
