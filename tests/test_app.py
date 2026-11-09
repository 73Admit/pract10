# -*- coding: utf-8 -*-
"""Модульные тесты ML-продукта (pytest).

Запуск из папки PR10:
    python -m pytest tests -v
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ml_core          # noqa: E402
import storage          # noqa: E402
import recommend        # noqa: E402


# --------------------------------------------------------------------------- #
#  Тест 1. Маскировка телефонов
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('text,expected_hidden', [
    ('Call me at 555-123-4567 please', '555-123-4567'),
    ('My number is (800) 555 0199', '(800) 555 0199'),
    ('Reach +1 212 555 7788 anytime', '212 555 7788'),
])
def test_mask_phone_hides_number(text, expected_hidden):
    masked = ml_core.mask_phone(text)
    assert expected_hidden not in masked
    assert ml_core.PHONE_MASK in masked


def test_mask_phone_keeps_normal_numbers():
    """Обычные числа в тексте не должны считаться телефоном."""
    text = 'I played 40 hours and paid 59 dollars'
    assert ml_core.mask_phone(text) == text


# --------------------------------------------------------------------------- #
#  Тест 2. Маскировка имени, почты и карты
# --------------------------------------------------------------------------- #

def test_mask_name_and_email_and_card():
    text = ('Hello, my name is John, write to john.doe@example.com, '
            'card 4111 1111 1111 1111')
    masked = ml_core.mask_pii(text)
    assert 'John' not in masked
    assert 'john.doe@example.com' not in masked
    assert '4111 1111 1111 1111' not in masked
    assert ml_core.NAME_MASK in masked
    assert ml_core.EMAIL_MASK in masked
    assert ml_core.CARD_MASK in masked


def test_mask_pii_does_not_break_plain_text():
    text = 'The controller works great and the battery lasts long'
    assert ml_core.mask_pii(text) == text


# --------------------------------------------------------------------------- #
#  Тест 3. Загрузка данных
# --------------------------------------------------------------------------- #

def test_load_data_structure():
    df = ml_core.load_data()
    for col in ('review_id', 'user_id', 'product', 'rating', 'text', 'date',
                'sentiment'):
        assert col in df.columns, f'нет столбца {col}'
    assert len(df) > 200, 'в датасете должно быть не менее 200 записей'
    assert df['rating'].between(1, 5).all()
    assert set(df['sentiment'].unique()) <= {0, 1}
    assert df['text'].isna().sum() == 0


# --------------------------------------------------------------------------- #
#  Тест 4. Предсказание тональности
# --------------------------------------------------------------------------- #

def test_predict_sentiment_positive_and_negative():
    model = ml_core.load_sentiment_model()
    if model is None:
        pytest.skip('модель не обучена: запустите train_models.py')
    pos = ml_core.predict_sentiment('Amazing product, works perfectly, I love it!')
    neg = ml_core.predict_sentiment('Terrible quality, broke after one day, waste of money')
    assert pos['positive'] is True
    assert neg['positive'] is False
    assert 0 <= pos['confidence'] <= 100
    assert pos['label'] == 'позитивный' and neg['label'] == 'негативный'


# --------------------------------------------------------------------------- #
#  Тест 5. Аналитические функции
# --------------------------------------------------------------------------- #

def test_top_products_and_negative_words():
    df = ml_core.load_data()
    top = ml_core.top_products(df, n=5)
    assert len(top) == 5
    assert top['доля_позитива'].is_monotonic_decreasing
    assert (top['доля_позитива'] <= 100).all()

    words = ml_core.frequent_negative_words(df, 5)
    assert len(words) == 5
    assert all(isinstance(w, str) and c > 0 for w, c in words)
    assert words[0][1] >= words[-1][1]


# --------------------------------------------------------------------------- #
#  Тест 6. Метрики качества
# --------------------------------------------------------------------------- #

def test_metrics_file_is_valid():
    path = os.path.join(ml_core.BASE_DIR, 'models', 'metrics.csv')
    if not os.path.exists(path):
        pytest.skip('метрики не рассчитаны: запустите train_models.py')
    m = pd.read_csv(path)
    assert len(m) == 10, 'пять моделей по два этапа'
    for col in ('Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'):
        assert m[col].between(0, 1).all(), f'метрика {col} вне диапазона 0..1'
    assert set(m['Этап']) == {'до подбора', 'после подбора'}


# --------------------------------------------------------------------------- #
#  Тест 7. Рекомендательная система
# --------------------------------------------------------------------------- #

def test_recommendations_for_new_user():
    df = ml_core.load_data()
    storage.init_db()
    recs, note = recommend.recommend('user_which_does_not_exist', df, n=5)
    assert len(recs) == 5
    assert isinstance(note, str) and note
    for r in recs:
        assert 0 <= r['positive'] <= 100
        assert 1 <= r['predicted'] <= 5


# --------------------------------------------------------------------------- #
#  Тест 8. База данных и журнал
# --------------------------------------------------------------------------- #

def test_storage_roundtrip_and_log():
    storage.init_db()
    review_id = storage.add_review('test_user', 'Test product', 5,
                                   'Call 555-123-4567', '[ТЕЛЕФОН СКРЫТ]', 1, 99.0)
    assert review_id > 0
    rows = storage.get_reviews('test_user', limit=5)
    assert rows and rows[0]['product'] == 'Test product'
    assert rows[0]['rating'] == 5

    storage.log_event('test_user', 'unit test', 'ok')
    lines = storage.read_log(10)
    assert any('unit test' in l for l in lines)
