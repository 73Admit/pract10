# -*- coding: utf-8 -*-
"""Ядро ML-продукта: загрузка данных, защита персональных данных,
предсказание тональности и аналитические функции."""
from __future__ import annotations

import os
import re
from collections import Counter

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, 'data', 'reviews.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
SENT_MODEL = os.path.join(MODEL_DIR, 'sentiment_model.pkl')

# --------------------------------------------------------------------------- #
#  Защита персональных данных
# --------------------------------------------------------------------------- #

PHONE_RE = re.compile(
    r'(?<!\d)(?:\+?\d{1,2}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)')
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
CARD_RE = re.compile(r'(?<!\d)(?:\d{4}[\s-]?){3}\d{4}(?!\d)')
NAME_RE = re.compile(
    r'\b(?:my name is|i am|i\'m|this is|меня зовут|я\s*—?)\s+([A-ZА-Я][a-zа-я]+)',
    re.IGNORECASE)

PHONE_MASK = '[ТЕЛЕФОН СКРЫТ]'
EMAIL_MASK = '[EMAIL СКРЫТ]'
CARD_MASK = '[КАРТА СКРЫТА]'
NAME_MASK = '[ИМЯ СКРЫТО]'


def mask_phone(text: str) -> str:
    """Заменяет телефонные номера на заглушку."""
    return PHONE_RE.sub(PHONE_MASK, text or '')


def mask_email(text: str) -> str:
    """Заменяет адреса электронной почты на заглушку."""
    return EMAIL_RE.sub(EMAIL_MASK, text or '')


def mask_card(text: str) -> str:
    """Заменяет номера банковских карт на заглушку."""
    return CARD_RE.sub(CARD_MASK, text or '')


def mask_name(text: str) -> str:
    """Заменяет имя, названное в тексте («my name is John»), на заглушку."""
    def repl(m: re.Match) -> str:
        return m.group(0).replace(m.group(1), NAME_MASK)
    return NAME_RE.sub(repl, text or '')


def mask_pii(text: str) -> str:
    """Полная маскировка персональных данных: карта, телефон, почта, имя."""
    text = text or ''
    text = mask_card(text)
    text = mask_phone(text)
    text = mask_email(text)
    text = mask_name(text)
    return text


# --------------------------------------------------------------------------- #
#  Загрузка данных
# --------------------------------------------------------------------------- #

def load_data(path: str = DATA_CSV, mask: bool = True) -> pd.DataFrame:
    """Загружает отзывы и при необходимости маскирует персональные данные."""
    df = pd.read_csv(path, parse_dates=['date'])
    df['text'] = df['text'].astype(str)
    if mask:
        df['text'] = df['text'].apply(mask_pii)
    df['sentiment'] = (df['rating'] >= 4).astype(int)
    return df


# --------------------------------------------------------------------------- #
#  Предсказание тональности
# --------------------------------------------------------------------------- #

_model_cache = {}


def load_sentiment_model(path: str = SENT_MODEL):
    if path not in _model_cache:
        _model_cache[path] = joblib.load(path) if os.path.exists(path) else None
    return _model_cache[path]


def predict_sentiment(text: str, model=None) -> dict:
    """Возвращает «позитивный»/«негативный» и процент уверенности модели."""
    model = model or load_sentiment_model()
    if model is None:
        raise RuntimeError('Модель не обучена: запустите train_models.py')
    clean = mask_pii(str(text))
    proba = model.predict_proba([clean])[0]
    label = int(np.argmax(proba))
    return {
        'label': 'позитивный' if label == 1 else 'негативный',
        'positive': label == 1,
        'confidence': round(float(proba[label]) * 100, 1),
        'p_positive': round(float(proba[1]) * 100, 1),
        'masked_text': clean,
    }


# --------------------------------------------------------------------------- #
#  Аналитика
# --------------------------------------------------------------------------- #

def top_products(df: pd.DataFrame, n: int = 10, min_reviews: int = 10) -> pd.DataFrame:
    """Топ товаров по доле позитивных отзывов."""
    g = (df.groupby('product')
           .agg(отзывов=('rating', 'size'),
                средняя_оценка=('rating', 'mean'),
                доля_позитива=('sentiment', 'mean'))
           .query('отзывов >= @min_reviews'))
    g['средняя_оценка'] = g['средняя_оценка'].round(2)
    g['доля_позитива'] = (g['доля_позитива'] * 100).round(1)
    return g.sort_values(['доля_позитива', 'отзывов'], ascending=False).head(n)


STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been',
    'to', 'of', 'in', 'on', 'for', 'with', 'at', 'by', 'from', 'it', 'its', 'this',
    'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'my', 'your',
    'me', 'him', 'her', 'them', 'us', 'as', 'so', 'if', 'then', 'than', 'too', 'very',
    'can', 'will', 'just', 'do', 'does', 'did', 'have', 'has', 'had', 'not', 'no',
    'up', 'out', 'about', 'into', 'over', 'after', 'all', 'any', 'more', 'most',
    'other', 'some', 'such', 'only', 'own', 'same', 'what', 'which', 'who', 'when',
    'where', 'how', 'there', 'here', 'would', 'could', 'should', 'get', 'got', 'one',
    'also', 'been', 'am', 'br', 'im', 'ive', 'dont', 'didnt', 'doesnt', 'cant', 'its',
    # служебные слова заглушек, которыми заменяются персональные данные
    'имя', 'скрыто', 'скрыта', 'телефон', 'карта', 'email',
}


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r'[a-zA-Zа-яА-ЯёЁ]{3,}', str(text).lower())
            if w not in STOP_WORDS]


def frequent_negative_words(df: pd.DataFrame, n: int = 5) -> list[tuple[str, int]]:
    """n самых частых слов в негативных отзывах."""
    neg = df[df['sentiment'] == 0]['text']
    counter: Counter = Counter()
    for t in neg:
        counter.update(tokenize(t))
    return counter.most_common(n)


def frequent_words(texts, n: int = 10) -> list[tuple[str, int]]:
    counter: Counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    return counter.most_common(n)


def product_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Полная статистика по товарам: средний рейтинг, число отзывов, доля позитива."""
    g = (df.groupby(['product', 'brand'])
           .agg(отзывов=('rating', 'size'),
                средний_рейтинг=('rating', 'mean'),
                доля_позитива=('sentiment', 'mean'),
                первый_отзыв=('date', 'min'),
                последний_отзыв=('date', 'max'))
           .reset_index())
    g['средний_рейтинг'] = g['средний_рейтинг'].round(2)
    g['доля_позитива'] = (g['доля_позитива'] * 100).round(1)
    return g.sort_values('отзывов', ascending=False)


def rating_distribution(df: pd.DataFrame) -> dict:
    return df['rating'].value_counts().sort_index().to_dict()


def reviews_by_day(df: pd.DataFrame) -> pd.Series:
    return df.set_index('date').resample('MS').size()
