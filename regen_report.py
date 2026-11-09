# -*- coding: utf-8 -*-
"""Пересборка сводного отчёта и описаний кластеров без повторного обучения моделей.

Используется, когда изменились только аналитические функции (например, список
стоп-слов), а обученные модели и эмбеддинги остались прежними.
"""
from __future__ import annotations

import json
import os

import pandas as pd

import ml_core
import train_models

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, 'models')


def main() -> None:
    df = ml_core.load_data()
    results = pd.read_csv(os.path.join(MODELS, 'metrics.csv'))
    with open(os.path.join(MODELS, 'summary.json'), encoding='utf-8') as f:
        summary = json.load(f)
    clusters = pd.read_csv(os.path.join(MODELS, 'clusters.csv'))

    merged = df.merge(clusters[['review_id', 'cluster_km']], on='review_id')
    cluster_info = []
    for c in sorted(merged['cluster_km'].unique()):
        part = merged[merged['cluster_km'] == c]
        words = ml_core.frequent_words(part['text'], 10)
        cluster_info.append({
            'cluster': int(c), 'size': int(len(part)),
            'mean_rating': round(float(part['rating'].mean()), 2),
            'positive_share': round(float(part['sentiment'].mean()) * 100, 1),
            'top_words': [w for w, _ in words],
            'examples': part['text'].head(5).str[:220].tolist(),
        })
        print(f"Кластер {c}: {len(part)} отзывов, средняя оценка "
              f"{part['rating'].mean():.2f}, позитива "
              f"{part['sentiment'].mean()*100:.1f} %, слова: "
              f"{', '.join(w for w, _ in words[:6])}")

    cl_table = pd.DataFrame(summary['clustering'])
    train_models.build_student_report(df, results, cl_table, cluster_info,
                                      summary['best_model'])

    summary['clusters'] = cluster_info
    summary['negative_words'] = ml_core.frequent_negative_words(df, 10)
    summary['top_products'] = (ml_core.top_products(df).reset_index()
                               .to_dict('records'))
    with open(os.path.join(MODELS, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print('summary.json и student_report.png обновлены')


if __name__ == '__main__':
    main()
