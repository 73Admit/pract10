# -*- coding: utf-8 -*-
"""Обучение и сравнение моделей классификации тональности, кластеризация отзывов.

Этапы:
1. загрузка данных с маскировкой персональных данных;
2. пять моделей классификации на TF-IDF с базовыми параметрами;
3. подбор гиперпараметров каждой модели через GridSearchCV (не менее трёх параметров);
4. сводная таблица метрик до и после подбора, ROC-кривые;
5. векторизация отзывов через SentenceTransformer;
6. кластеризация KMeans и DBSCAN, визуализация через PCA, топ-10 слов
   и средняя тональность каждого кластера;
7. сводный отчёт student_report.png.
"""
from __future__ import annotations

import json
import os
import time
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import ml_core

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, 'models')
IMG = os.path.join(BASE, 'img')
STATIC = os.path.join(BASE, 'static')
SEED = 42

os.makedirs(MODELS, exist_ok=True)
os.makedirs(IMG, exist_ok=True)
os.makedirs(STATIC, exist_ok=True)
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 110


# --------------------------------------------------------------------------- #
def tfidf_pipe(clf):
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=8000, ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True)),
        ('clf', clf),
    ])


def evaluate(name, model, X_test, y_test, stage):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        'Модель': name, 'Этап': stage,
        'Accuracy': round(accuracy_score(y_test, pred), 4),
        'Precision': round(precision_score(y_test, pred, zero_division=0), 4),
        'Recall': round(recall_score(y_test, pred, zero_division=0), 4),
        'F1': round(f1_score(y_test, pred, zero_division=0), 4),
        'ROC-AUC': round(roc_auc_score(y_test, proba), 4),
    }


def main() -> None:
    t0 = time.time()
    print('=== 1. Загрузка данных ===')
    df = ml_core.load_data()
    print('Отзывов:', len(df), '| товаров:', df['product'].nunique(),
          '| пользователей:', df['user_id'].nunique())
    print('Доля позитивных отзывов:', round(df['sentiment'].mean(), 4))

    X = df['text'].values
    y = df['sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    print('Обучающая выборка:', len(X_train), '| тестовая:', len(X_test))

    # ------------------------------------------------------------------ #
    print('\n=== 2. Пять моделей с базовыми параметрами ===')
    base_models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=SEED),
        'RandomForest': RandomForestClassifier(random_state=SEED, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=SEED, eval_metric='logloss',
                                 verbosity=0, n_jobs=-1),
        'MLPClassifier': MLPClassifier(random_state=SEED, max_iter=300),
        'CatBoost': CatBoostClassifier(random_state=SEED, verbose=0),
    }
    rows, fitted_base = [], {}
    for name, clf in base_models.items():
        t = time.time()
        pipe = tfidf_pipe(clf).fit(X_train, y_train)
        fitted_base[name] = pipe
        r = evaluate(name, pipe, X_test, y_test, 'до подбора')
        r['Время, с'] = round(time.time() - t, 1)
        rows.append(r)
        print(f"  {name:20s} F1={r['F1']:.4f}  ROC-AUC={r['ROC-AUC']:.4f}  "
              f"({r['Время, с']} с)")

    # ------------------------------------------------------------------ #
    print('\n=== 3. Подбор гиперпараметров (GridSearchCV, >=3 параметра) ===')
    grids = {
        'LogisticRegression': (LogisticRegression(max_iter=1000, random_state=SEED), {
            'clf__C': [0.1, 1, 10],
            'clf__penalty': ['l2'],
            'clf__class_weight': [None, 'balanced'],
            'clf__solver': ['lbfgs', 'liblinear'],
        }),
        'RandomForest': (RandomForestClassifier(random_state=SEED, n_jobs=-1), {
            'clf__n_estimators': [200, 400],
            'clf__max_depth': [None, 20],
            'clf__min_samples_leaf': [1, 3],
            'clf__class_weight': [None, 'balanced'],
        }),
        'XGBoost': (XGBClassifier(random_state=SEED, eval_metric='logloss',
                                  verbosity=0, n_jobs=-1), {
            'clf__n_estimators': [200, 400],
            'clf__max_depth': [3, 6],
            'clf__learning_rate': [0.1, 0.3],
            'clf__subsample': [0.8, 1.0],
        }),
        'MLPClassifier': (MLPClassifier(random_state=SEED, max_iter=300), {
            'clf__hidden_layer_sizes': [(64,), (128, 64)],
            'clf__alpha': [1e-4, 1e-3],
            'clf__learning_rate_init': [1e-3, 3e-3],
        }),
        'CatBoost': (CatBoostClassifier(random_state=SEED, verbose=0), {
            'clf__iterations': [300, 600],
            'clf__depth': [4, 6],
            'clf__learning_rate': [0.05, 0.1],
        }),
    }

    best_params, fitted_tuned = {}, {}
    for name, (clf, grid) in grids.items():
        t = time.time()
        gs = GridSearchCV(tfidf_pipe(clf), grid, scoring='f1', cv=3, n_jobs=-1)
        gs.fit(X_train, y_train)
        fitted_tuned[name] = gs.best_estimator_
        best_params[name] = {k.replace('clf__', ''): v for k, v in gs.best_params_.items()}
        r = evaluate(name, gs.best_estimator_, X_test, y_test, 'после подбора')
        r['Время, с'] = round(time.time() - t, 1)
        rows.append(r)
        print(f"  {name:20s} F1={r['F1']:.4f}  ROC-AUC={r['ROC-AUC']:.4f}  "
              f"({r['Время, с']} с)  параметры: {best_params[name]}")

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(MODELS, 'metrics.csv'), index=False, encoding='utf-8')

    pivot = results.pivot(index='Модель', columns='Этап',
                          values=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'])
    print('\n=== Сводная таблица метрик до и после подбора ===')
    print(pivot.round(4).to_string())

    best_name = (results[results['Этап'] == 'после подбора']
                 .sort_values('F1', ascending=False).iloc[0]['Модель'])
    best_model = fitted_tuned[best_name]
    joblib.dump(best_model, os.path.join(MODELS, 'sentiment_model.pkl'))
    print(f'\nЛучшая модель: {best_name}; сохранена в models/sentiment_model.pkl')

    # ------------------------------------------------------------------ #
    print('\n=== 4. ROC-кривые ===')
    plt.figure(figsize=(8, 6.5))
    for name, model in fitted_tuned.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, linewidth=2,
                 label=f'{name} (AUC = {roc_auc_score(y_test, proba):.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='случайная модель')
    plt.xlabel('Доля ложноположительных (FPR)')
    plt.ylabel('Доля истинноположительных (TPR)')
    plt.title('ROC-кривые моделей после подбора гиперпараметров')
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG, 'roc_curves.png'), bbox_inches='tight')
    plt.savefig(os.path.join(STATIC, 'roc_curves.png'), bbox_inches='tight')
    plt.close()

    cm = confusion_matrix(y_test, best_model.predict(X_test))
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['негатив', 'позитив'], yticklabels=['негатив', 'позитив'])
    plt.title(f'Матрица ошибок лучшей модели ({best_name})')
    plt.xlabel('Предсказание'); plt.ylabel('Факт')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------ #
    print('\n=== 5. Векторизация SentenceTransformer ===')
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    texts = df['text'].tolist()
    emb = st_model.encode(texts, batch_size=64, show_progress_bar=False,
                          normalize_embeddings=True)
    np.save(os.path.join(MODELS, 'embeddings.npy'), emb)
    print('Размер матрицы эмбеддингов:', emb.shape)

    # ------------------------------------------------------------------ #
    print('\n=== 6. Кластеризация отзывов ===')
    km = KMeans(n_clusters=5, random_state=SEED, n_init=10)
    lab_km = km.fit_predict(emb)
    db = DBSCAN(eps=0.9, min_samples=25, metric='euclidean')
    lab_db = db.fit_predict(emb)

    def cl_metrics(name, labels):
        mask = labels != -1
        k = len(set(labels[mask]))
        if k < 2:
            return {'Модель': name, 'Кластеров': k, 'Шум': int((labels == -1).sum()),
                    'Silhouette': None, 'Davies-Bouldin': None, 'Calinski-Harabasz': None}
        return {'Модель': name, 'Кластеров': k, 'Шум': int((labels == -1).sum()),
                'Silhouette': round(silhouette_score(emb[mask], labels[mask]), 4),
                'Davies-Bouldin': round(davies_bouldin_score(emb[mask], labels[mask]), 4),
                'Calinski-Harabasz': round(calinski_harabasz_score(emb[mask], labels[mask]), 1)}

    cl_table = pd.DataFrame([cl_metrics('KMeans (k=5)', lab_km),
                             cl_metrics('DBSCAN (eps=0.9)', lab_db)])
    print(cl_table.to_string(index=False))

    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(emb)
    print('Две компоненты PCA объясняют',
          round(pca.explained_variance_ratio_.sum() * 100, 2), '% дисперсии')

    df_cl = df.copy()
    df_cl['cluster_km'] = lab_km
    df_cl['cluster_db'] = lab_db
    df_cl['pc1'] = coords[:, 0]
    df_cl['pc2'] = coords[:, 1]

    cluster_info = []
    for c in sorted(set(lab_km)):
        part = df_cl[df_cl['cluster_km'] == c]
        words = ml_core.frequent_words(part['text'], 10)
        cluster_info.append({
            'cluster': int(c), 'size': int(len(part)),
            'mean_rating': round(float(part['rating'].mean()), 2),
            'positive_share': round(float(part['sentiment'].mean()) * 100, 1),
            'top_words': [w for w, _ in words],
            'examples': part['text'].head(5).str[:220].tolist(),
        })
        print(f"  Кластер {c}: {len(part)} отзывов, средняя оценка "
              f"{part['rating'].mean():.2f}, позитива "
              f"{part['sentiment'].mean()*100:.1f} %, слова: "
              f"{', '.join(w for w, _ in words[:6])}")

    df_cl[['review_id', 'product', 'rating', 'sentiment', 'cluster_km', 'cluster_db',
           'pc1', 'pc2']].to_csv(os.path.join(MODELS, 'clusters.csv'),
                                 index=False, encoding='utf-8')

    import plotly.express as px
    fig = px.scatter(df_cl, x='pc1', y='pc2', color=df_cl['cluster_km'].astype(str),
                     hover_data={'product': True, 'rating': True, 'pc1': False,
                                 'pc2': False},
                     labels={'color': 'кластер'},
                     title='Кластеры отзывов в пространстве двух главных компонент')
    fig.update_traces(marker=dict(size=5, opacity=0.65))
    fig.update_layout(height=620, legend_title_text='Кластер')
    fig.write_html(os.path.join(STATIC, 'clusters.html'), include_plotlyjs='cdn')
    fig.write_image(os.path.join(IMG, 'clusters_plotly.png'), width=1100, height=620,
                    scale=2) if _can_write_image() else None

    plt.figure(figsize=(9.5, 6.5))
    for c in sorted(set(lab_km)):
        m = lab_km == c
        plt.scatter(coords[m, 0], coords[m, 1], s=7, alpha=.6, label=f'кластер {c}')
    plt.title('Кластеры отзывов (KMeans, эмбеддинги SentenceTransformer, проекция PCA)')
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG, 'clusters.png'), bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------ #
    print('\n=== 7. Сводный отчёт student_report.png ===')
    build_student_report(df, results, cl_table, cluster_info, best_name)

    summary = {
        'rows': int(len(df)),
        'products': int(df['product'].nunique()),
        'users': int(df['user_id'].nunique()),
        'positive_share': round(float(df['sentiment'].mean()), 4),
        'best_model': best_name,
        'best_params': best_params,
        'metrics': results.to_dict('records'),
        'clustering': cl_table.to_dict('records'),
        'clusters': cluster_info,
        'pca_variance': round(float(pca.explained_variance_ratio_.sum()), 4),
        'embedding_dim': int(emb.shape[1]),
        'top_products': ml_core.top_products(df).reset_index().to_dict('records'),
        'negative_words': ml_core.frequent_negative_words(df, 10),
    }
    with open(os.path.join(MODELS, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    print(f'\nГотово за {time.time() - t0:.0f} с. Артефакты в папке models/, '
          f'графики в img/ и static/')


def _can_write_image() -> bool:
    try:
        import kaleido  # noqa: F401
        return True
    except Exception:
        return False


def build_student_report(df, results, cl_table, cluster_info, best_name) -> None:
    """Сводный графический отчёт по требованию задания — student_report.png."""
    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    counts = df['rating'].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color='#4c72b0')
    for x, v in zip(counts.index, counts.values):
        ax.text(x, v, str(v), ha='center', va='bottom', fontsize=9)
    ax.set_title('Распределение оценок'); ax.set_xlabel('Оценка'); ax.set_ylabel('Отзывов')

    ax = fig.add_subplot(gs[0, 1])
    ts = df.set_index('date').resample('QE').size()
    ax.plot(ts.index, ts.values, linewidth=2, color='#c44e52')
    ax.set_title('Динамика количества отзывов по кварталам')
    ax.set_xlabel('Дата'); ax.set_ylabel('Отзывов')

    ax = fig.add_subplot(gs[1, 0])
    tp = ml_core.top_products(df, 10).sort_values('доля_позитива')
    ax.barh([p[:38] for p in tp.index], tp['доля_позитива'], color='#55a868')
    ax.set_title('Топ-10 товаров по доле позитивных отзывов')
    ax.set_xlabel('Доля позитива, %')

    ax = fig.add_subplot(gs[1, 1])
    words = ml_core.frequent_negative_words(df, 10)
    ax.barh([w for w, _ in words][::-1], [c for _, c in words][::-1], color='#c44e52')
    ax.set_title('Частые слова в негативных отзывах')
    ax.set_xlabel('Количество упоминаний')

    ax = fig.add_subplot(gs[2, 0])
    after = results[results['Этап'] == 'после подбора'].set_index('Модель')
    before = results[results['Этап'] == 'до подбора'].set_index('Модель')
    idx = np.arange(len(after))
    ax.bar(idx - 0.2, before.loc[after.index, 'F1'], width=0.4, label='до подбора')
    ax.bar(idx + 0.2, after['F1'], width=0.4, label='после подбора')
    ax.set_xticks(idx); ax.set_xticklabels(after.index, rotation=20, ha='right')
    ax.set_ylim(0, 1.05); ax.set_title('F1-мера моделей до и после подбора параметров')
    ax.legend()

    ax = fig.add_subplot(gs[2, 1])
    sizes = [c['size'] for c in cluster_info]
    labels = [f"кластер {c['cluster']}\n{c['positive_share']}% позитива" for c in cluster_info]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
           textprops={'fontsize': 9})
    ax.set_title('Распределение отзывов по кластерам (KMeans)')

    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')
    lines = [
        f'ML-продукт для анализа отзывов интернет-магазина GameStop',
        f'Отзывов: {len(df)}   товаров: {df["product"].nunique()}   '
        f'пользователей: {df["user_id"].nunique()}   '
        f'период: {df["date"].min():%Y-%m-%d} — {df["date"].max():%Y-%m-%d}',
        f'Доля позитивных отзывов: {df["sentiment"].mean()*100:.1f} %   '
        f'средняя оценка: {df["rating"].mean():.2f}',
        '',
        'Метрики моделей после подбора гиперпараметров:',
    ]
    for _, r in after.reset_index().iterrows():
        lines.append(f'   {r["Модель"]:<20} accuracy {r["Accuracy"]:.4f}   '
                     f'precision {r["Precision"]:.4f}   recall {r["Recall"]:.4f}   '
                     f'F1 {r["F1"]:.4f}   ROC-AUC {r["ROC-AUC"]:.4f}')
    lines += ['', f'Лучшая модель: {best_name}', '', 'Кластеризация отзывов:']
    for _, r in cl_table.iterrows():
        lines.append(f'   {r["Модель"]:<20} кластеров {r["Кластеров"]}   '
                     f'шум {r["Шум"]}   silhouette {r["Silhouette"]}   '
                     f'davies-bouldin {r["Davies-Bouldin"]}')
    ax.text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=11)

    out = os.path.join(BASE, 'student_report.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('Сохранён', out)


if __name__ == '__main__':
    main()
