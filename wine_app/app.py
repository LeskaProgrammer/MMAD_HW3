import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


st.set_page_config(page_title='оценка вина', layout='wide')
st.title('оценка вина')

model = load_model()
test = load_test()
X_test = test.drop(columns='target')
y_test = test['target']
features = X_test.columns.tolist()
config = json.loads((APP_DIR / 'config.json').read_text())

threshold = st.sidebar.slider(
    'порог хорошего вина', 0.0, 1.0, float(config['threshold']),
    step=0.01, format='%.3f',
)
st.sidebar.caption('один порог для всех вкладок')

one_tab, metrics_tab, batch_tab, coef_tab = st.tabs([
    'одно вино', 'модельные метрики', 'загрузка csv', 'коэффициенты',
])


with one_tab:
    st.write('введите характеристики вина')
    inputs = {}
    columns = st.columns(2)
    for i, feature in enumerate(features):
        with columns[i % 2]:
            if feature == 'is_red':
                color = st.selectbox('цвет вина', ['белое', 'красное'])
                inputs[feature] = int(color == 'красное')
            else:
                inputs[feature] = st.number_input(
                    feature,
                    min_value=float(X_test[feature].min()),
                    max_value=float(X_test[feature].max()),
                    value=float(X_test[feature].median()),
                    format='%.4f',
                )

    wine = pd.DataFrame([inputs], columns=features)
    probability = float(model.predict_proba(wine)[0, 1])
    st.metric('вероятность хорошего вина', f'{probability:.1%}')
    st.write('решение: хорошее вино' if probability >= threshold
             else 'решение: не включать в подборку')


with metrics_tab:
    proba = test_proba(model, X_test)
    pred = proba >= threshold
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)

    columns = st.columns(3)
    for column, name, value in zip(columns, ['precision', 'recall', 'f1'],
                                    [precision, recall, f1]):
        column.metric(name, f'{value:.3f}')

    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=['остальные', 'хорошие']).plot(ax=ax)
    ax.set_xlabel('предсказанный класс')
    ax.set_ylabel('настоящий класс')
    ax.grid(False)
    fig.tight_layout()
    st.pyplot(fig, width=500)
    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    metrics.RocCurveDisplay.from_predictions(y_test, proba, name='модель', ax=axes[0])
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].scatter(fp / (fp + tn), recall, label='текущий порог', color='red', zorder=3)
    metrics.PrecisionRecallDisplay.from_predictions(y_test, proba, name='модель', ax=axes[1])
    axes[1].axhline(y_test.mean(), ls='--', color='gray')
    axes[1].scatter(recall, precision, label='текущий порог', color='red', zorder=3)
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    st.pyplot(fig, width=1000)
    plt.close(fig)

    st.write('если повысить порог, подборка станет меньше и отбор строже, '
             'но часть хороших вин будет пропущена. если снизить порог, '
             'вин в подборке станет больше, вместе с ними могут попасть посредственные')


with batch_tab:
    st.write('загрузите csv с теми же 12 признаками, что используются на вкладке одного вина')
    uploaded = st.file_uploader('файл с признаками вин', type='csv')
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            missing = [feature for feature in features if feature not in batch.columns]
            if missing:
                st.error('не хватает столбцов: ' + ', '.join(missing))
            elif batch.empty:
                st.error('в файле нет строк с винами')
            else:
                batch_features = batch[features].apply(pd.to_numeric, errors='raise')
                if not np.isfinite(batch_features.to_numpy()).all():
                    st.error('в признаках есть пропуски или бесконечные значения')
                else:
                    result = batch.copy()
                    result['probability'] = model.predict_proba(batch_features)[:, 1]
                    result['prediction'] = (result['probability'] >= threshold).astype(int)
                    st.dataframe(result)
                    st.download_button(
                        'скачать предсказания', result.to_csv(index=False).encode('utf-8'),
                        file_name='predictions.csv', mime='text/csv',
                    )
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError):
            st.error('не удалось прочитать csv: проверьте разделитель и числовые значения признаков')


with coef_tab:
    coefficients = pd.Series(model.steps[-1][1].coef_.ravel(), index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    coefficients.plot.barh(ax=ax)
    ax.axvline(0, color='gray', lw=1)
    ax.set_xlabel('коэффициент после масштабирования')
    fig.tight_layout()
    st.pyplot(fig, width=750)
    plt.close(fig)

    st.write('сильнее всего за хорошее вино: ' + ', '.join(coefficients.tail(3).index[::-1]))
    st.write('сильнее всего против: ' + ', '.join(coefficients.head(3).index))
