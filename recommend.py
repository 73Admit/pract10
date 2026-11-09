# -*- coding: utf-8 -*-
"""Персонализированные рекомендации товаров.

Основной метод — user-based collaborative filtering: по матрице «пользователь × товар»
ищутся похожие пользователи (косинусная близость профилей оценок), и товары,
которые они оценили высоко, предлагаются текущему пользователю.

Если истории мало (меньше двух оценок) или похожих пользователей не нашлось,
система возвращается к базовой стратегии: товары с наибольшей долей позитивных
отзывов среди тех, что пользователь ещё не оценивал.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import storage

MIN_HISTORY = 2
MIN_REVIEWS_PRODUCT = 10


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Матрица «пользователь × товар» из исторических и накопленных оценок."""
    hist = df[['user_id', 'product', 'rating']]
    web = pd.DataFrame(storage.user_ratings())
    data = pd.concat([hist, web], ignore_index=True) if len(web) else hist
    return data.pivot_table(index='user_id', columns='product', values='rating',
                            aggfunc='mean')


def popular_products(df: pd.DataFrame, exclude: set[str], n: int) -> list[dict]:
    g = (df.groupby('product')
           .agg(reviews=('rating', 'size'), rating=('rating', 'mean'),
                positive=('sentiment', 'mean'))
           .query('reviews >= @MIN_REVIEWS_PRODUCT'))
    g = g[~g.index.isin(exclude)]
    g = g.sort_values(['positive', 'reviews'], ascending=False).head(n)
    return [{'product': p, 'predicted': round(r['rating'], 2),
             'reviews': int(r['reviews']),
             'positive': round(r['positive'] * 100, 1),
             'reason': 'высокая доля позитивных отзывов'}
            for p, r in g.iterrows()]


def recommend(user_id: str, df: pd.DataFrame, n: int = 5) -> tuple[list[dict], str]:
    """Возвращает список рекомендаций и пояснение, каким методом они получены."""
    matrix = build_matrix(df)
    if user_id not in matrix.index:
        return (popular_products(df, set(), n),
                'История пуста: показаны товары с наибольшей долей позитивных отзывов. '
                'Оставьте две-три оценки, и рекомендации станут персональными.')

    user_row = matrix.loc[user_id]
    rated = set(user_row.dropna().index)
    if len(rated) < MIN_HISTORY:
        return (popular_products(df, rated, n),
                f'В вашей истории {len(rated)} оценка(и). Для коллаборативной '
                f'фильтрации нужно минимум {MIN_HISTORY}; пока показаны лучшие товары '
                f'магазина.')

    filled = matrix.fillna(0)
    sim = cosine_similarity(filled.loc[[user_id]], filled)[0]
    sim_series = pd.Series(sim, index=matrix.index).drop(user_id)
    neighbours = sim_series[sim_series > 0].sort_values(ascending=False).head(50)

    if neighbours.empty:
        return (popular_products(df, rated, n),
                'Похожих пользователей не нашлось — показаны лучшие товары магазина.')

    scores: dict[str, float] = {}
    weights: dict[str, float] = {}
    for other, weight in neighbours.items():
        row = matrix.loc[other].dropna()
        for product, rating in row.items():
            if product in rated:
                continue
            scores[product] = scores.get(product, 0) + weight * rating
            weights[product] = weights.get(product, 0) + weight

    if not scores:
        return (popular_products(df, rated, n),
                'Похожие пользователи не оценивали новых товаров — показаны лучшие '
                'товары магазина.')

    pred = {p: scores[p] / weights[p] for p in scores}
    top = sorted(pred.items(), key=lambda kv: -kv[1])[:n]
    agg = df.groupby('product').agg(reviews=('rating', 'size'),
                                    positive=('sentiment', 'mean'))
    out = []
    for product, value in top:
        info = agg.loc[product] if product in agg.index else None
        out.append({
            'product': product,
            'predicted': round(float(value), 2),
            'reviews': int(info['reviews']) if info is not None else 0,
            'positive': round(float(info['positive']) * 100, 1) if info is not None else 0,
            'reason': f'похожие пользователи ({len(neighbours)}) оценили высоко',
        })
    note = (f'Коллаборативная фильтрация: найдено {len(neighbours)} похожих '
            f'пользователей, ваша история — {len(rated)} оценок.')
    return out, note
