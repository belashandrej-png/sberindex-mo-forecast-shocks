"""Протокол бэктеста и метрики качества (MAE, WAPE, R2, тест Уилкоксона)."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .config import get


def mae(y_true, y_pred) -> float:
    """Средняя абсолютная ошибка — основная метрика конкурса."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def wape(y_true, y_pred) -> float:
    """Взвешенная абсолютная процентная ошибка (масштабно-инвариантна)."""
    y_true = np.asarray(y_true)
    return float(np.abs(y_true - y_pred).sum() / y_true.sum())


def r2(y_true, y_pred) -> float:
    """Коэффициент детерминации на pooled-выборке горизонта."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())


def metrics_by_horizon(preds: pd.DataFrame,
                       horizons: list[int] | None = None) -> pd.DataFrame:
    """Таблица метрик по горизонтам: horizon, MAE, WAPE, R2."""
    horizons = horizons or get("backtest.horizons", [1, 3, 6, 12])
    rows = []
    for h in horizons:
        d = preds[preds.h == h]
        rows.append([h, mae(d.y_true, d.y_pred), wape(d.y_true, d.y_pred),
                     r2(d.y_true, d.y_pred)])
    return pd.DataFrame(rows, columns=["horizon", "MAE", "WAPE", "R2"]).round(3)


def wilcoxon_compare(preds_a: pd.DataFrame, preds_b: pd.DataFrame,
                     horizons: list[int] | None = None) -> pd.DataFrame:
    """Парный тест Уилкоксона на абсолютных ошибках двух моделей по МО.

    Returns: таблица horizon, p_value, share_mo_a_better
             (доля МО, где модель A ошибается меньше модели B).
    """
    horizons = horizons or get("backtest.horizons", [1, 3, 6, 12])
    rows = []
    for h in horizons:
        a = preds_a[preds_a.h == h].set_index("mo")
        b = preds_b[preds_b.h == h].set_index("mo")
        ea = (a.y_true - a.y_pred).abs()
        eb = (b.y_true - b.y_pred).abs()
        com = ea.index.intersection(eb.index)
        _, p = wilcoxon(ea[com], eb[com])
        rows.append([h, float(p), float((ea[com] < eb[com]).mean())])
    return pd.DataFrame(rows, columns=["horizon", "p_value", "share_mo_a_better"])


def last_block_cutoff(T: int, h: int) -> int:
    """Индекс последнего обучающего месяца при протоколе last-block."""
    return T - 1 - h
