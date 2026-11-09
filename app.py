# -*- coding: utf-8 -*-
"""Веб-приложение ML-продукта для анализа отзывов (Flask)."""
from __future__ import annotations

import json
import os
import uuid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from flask import (Flask, render_template, request, session, redirect,
                   url_for, flash)
from wordcloud import WordCloud

import ml_core
import recommend
import storage

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, 'static')
MODELS = os.path.join(BASE, 'models')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'reviews-ml-pr10-secret')

DF = ml_core.load_data()
storage.init_db()
log = storage.setup_logger()

with open(os.path.join(MODELS, 'summary.json'), encoding='utf-8') as f:
    SUMMARY = json.load(f)
METRICS = pd.read_csv(os.path.join(MODELS, 'metrics.csv'))
CLUSTERS = pd.read_csv(os.path.join(MODELS, 'clusters.csv'))


def current_user() -> str:
    if 'user_id' not in session:
        session['user_id'] = 'web_' + uuid.uuid4().hex[:8]
    return session['user_id']


@app.before_request
def log_request():
    if request.endpoint not in (None, 'static'):
        storage.log_event(session.get('user_id', 'anonymous'),
                          f'{request.method} {request.path}',
                          'запрос принят')


def build_charts() -> None:
    """Готовит графики для страницы статистики (строятся один раз)."""
    dist = os.path.join(STATIC, 'rating_dist.png')
    if not os.path.exists(dist):
        counts = DF['rating'].value_counts().sort_index()
        plt.figure(figsize=(6, 4))
        plt.bar(counts.index, counts.values, color='#4c72b0')
        for x, v in zip(counts.index, counts.values):
            plt.text(x, v, str(v), ha='center', va='bottom', fontsize=9)
        plt.title('Распределение оценок'); plt.xlabel('Оценка'); plt.ylabel('Отзывов')
        plt.tight_layout(); plt.savefig(dist); plt.close()

    dyn = os.path.join(STATIC, 'dynamics.png')
    if not os.path.exists(dyn):
        ts = DF.set_index('date').resample('QE').size()
        plt.figure(figsize=(7, 4))
        plt.plot(ts.index, ts.values, linewidth=2, color='#c44e52')
        plt.title('Динамика количества отзывов'); plt.xlabel('Дата')
        plt.ylabel('Отзывов')
        plt.tight_layout(); plt.savefig(dyn); plt.close()

    wc = os.path.join(STATIC, 'wordcloud.png')
    if not os.path.exists(wc):
        words = ' '.join(' '.join(ml_core.tokenize(t)) for t in DF['text'])
        cloud = WordCloud(width=1100, height=520, background_color='white',
                          colormap='viridis').generate(words)
        cloud.to_file(wc)

    wcn = os.path.join(STATIC, 'wordcloud_neg.png')
    if not os.path.exists(wcn):
        neg = DF[DF['sentiment'] == 0]['text']
        words = ' '.join(' '.join(ml_core.tokenize(t)) for t in neg)
        cloud = WordCloud(width=1100, height=520, background_color='white',
                          colormap='Reds').generate(words)
        cloud.to_file(wcn)


@app.route('/', methods=['GET', 'POST'])
def index():
    user = current_user()
    prediction = None
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        rating = int(request.form.get('rating', 5))
        product = request.form.get('product', '').strip() or 'Не указан'
        if text:
            prediction = ml_core.predict_sentiment(text)
            storage.add_review(user, product, rating, text,
                               prediction['masked_text'],
                               int(prediction['positive']), prediction['confidence'])
            storage.log_event(user, 'POST /review',
                              f"товар={product} оценка={rating} "
                              f"тональность={prediction['label']} "
                              f"уверенность={prediction['confidence']}%")
            flash('Отзыв сохранён в базе данных')
    recs, note = recommend.recommend(user, DF, 5)
    return render_template('index.html', user=user, prediction=prediction,
                           recs=recs, note=note,
                           products=sorted(DF['product'].unique())[:60],
                           my_reviews=storage.get_reviews(user, 10),
                           summary=SUMMARY)


@app.route('/stats')
def stats():
    build_charts()
    top = ml_core.top_products(DF, 10).reset_index()
    neg_words = ml_core.frequent_negative_words(DF, 10)
    return render_template('stats.html', user=current_user(), top=top,
                           neg_words=neg_words, db=storage.stats(),
                           summary=SUMMARY)


@app.route('/products')
def products():
    table = ml_core.product_stats(DF)
    return render_template('products.html', user=current_user(),
                           table=table.head(60), total=len(table))


@app.route('/models')
def models():
    pivot = METRICS.pivot(index='Модель', columns='Этап',
                          values=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'])
    return render_template('models.html', user=current_user(),
                           metrics=METRICS, pivot=pivot, summary=SUMMARY)


@app.route('/clusters')
def clusters():
    chosen = request.args.get('c', type=int)
    examples = []
    if chosen is not None:
        ids = CLUSTERS[CLUSTERS['cluster_km'] == chosen]['review_id'].head(10)
        examples = DF[DF['review_id'].isin(ids)][
            ['product', 'rating', 'text']].to_dict('records')
    return render_template('clusters.html', user=current_user(),
                           info=SUMMARY['clusters'], chosen=chosen,
                           examples=examples, summary=SUMMARY)


@app.route('/logs')
def logs():
    return render_template('logs.html', user=current_user(),
                           lines=storage.read_log(50))


if __name__ == '__main__':
    app.run(debug=False, port=5001)
