# -*- coding: utf-8 -*-
"""Хранилище отзывов пользователей веб-приложения (SQLite) и журнал запросов."""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'reviews.db')
LOG_PATH = os.path.join(BASE_DIR, 'app.log')

SCHEMA = """
CREATE TABLE IF NOT EXISTS user_reviews (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT    NOT NULL,
    product    TEXT    NOT NULL,
    rating     INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    text       TEXT    NOT NULL,
    masked     TEXT,
    sentiment  INTEGER,
    confidence REAL,
    created    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ur_user ON user_reviews(user_id);
CREATE INDEX IF NOT EXISTS idx_ur_product ON user_reviews(product);
"""


# --------------------------------------------------------------------------- #
#  Журналирование
# --------------------------------------------------------------------------- #

def setup_logger() -> logging.Logger:
    """Все запросы к веб-приложению пишутся в app.log."""
    logger = logging.getLogger('reviews-app')
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH, encoding='utf-8')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    return logger


def log_event(user_id: str, action: str, result: str = '') -> None:
    setup_logger().info('user=%s | action=%s | result=%s', user_id, action, result)


def read_log(limit: int = 50) -> list[str]:
    """Последние записи журнала — для страницы администратора."""
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, encoding='utf-8') as f:
        lines = f.readlines()
    return [l.rstrip('\n') for l in lines[-limit:]][::-1]


# --------------------------------------------------------------------------- #
#  База данных
# --------------------------------------------------------------------------- #

def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with connect() as con:
        con.executescript(SCHEMA)


def add_review(user_id: str, product: str, rating: int, text: str,
               masked: str = '', sentiment: int | None = None,
               confidence: float | None = None) -> int:
    with connect() as con:
        cur = con.execute(
            'INSERT INTO user_reviews (user_id, product, rating, text, masked,'
            ' sentiment, confidence, created) VALUES (?,?,?,?,?,?,?,?)',
            (user_id, product, int(rating), text, masked, sentiment, confidence,
             datetime.now().isoformat(timespec='seconds')))
        return int(cur.lastrowid)


def get_reviews(user_id: str | None = None, limit: int = 100) -> list[dict]:
    sql = 'SELECT * FROM user_reviews'
    args: list = []
    if user_id:
        sql += ' WHERE user_id = ?'
        args.append(user_id)
    sql += ' ORDER BY id DESC LIMIT ?'
    args.append(limit)
    with connect() as con:
        return [dict(r) for r in con.execute(sql, args)]


def user_ratings() -> list[dict]:
    """Все оценки из веб-приложения — для коллаборативной фильтрации."""
    with connect() as con:
        return [dict(r) for r in con.execute(
            'SELECT user_id, product, rating FROM user_reviews')]


def stats() -> dict:
    with connect() as con:
        total = con.execute('SELECT COUNT(*) c FROM user_reviews').fetchone()['c']
        users = con.execute(
            'SELECT COUNT(DISTINCT user_id) c FROM user_reviews').fetchone()['c']
        avg = con.execute('SELECT AVG(rating) a FROM user_reviews').fetchone()['a']
    return {'total': total, 'users': users, 'avg_rating': round(avg, 2) if avg else None}


if __name__ == '__main__':
    init_db()
    print('База данных создана:', DB_PATH)
