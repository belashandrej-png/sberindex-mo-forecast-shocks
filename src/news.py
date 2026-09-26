"""Пайплайн согласования новостных данных с тратами.

Схема: событие (дата, impact) -> score с экспоненциальным затуханием ->
оценка лага опережения перебором по нормированной кросс-корреляции ->
event study (CAR на сезонно-скорректированных остатках).
"""
import numpy as np
import pandas as pd

from .config import get


def build_score(events: pd.DataFrame, date_index: pd.Series,
                lead_override: int | None = None,
                decay: float | None = None) -> np.ndarray:
    """Новостной признак: score[t] = sum impact * exp(-decay * d) по окну 4 мес.

    Args:
        events: DataFrame с колонками date, impact, true_lead (или kind).
        date_index: Series дат ряда (возрастание).
        lead_override: принудительный лаг (для grid search); иначе true_lead.
        decay: параметр затухания (по умолчанию из config: news.decay).
    """
    decay = decay if decay is not None else get("news.decay", 0.7)
    n = len(date_index)
    score = np.zeros(n)
    for _, ev in events.iterrows():
        lead = ev["true_lead"] if lead_override is None else lead_override
        onset = date_index.index[date_index >= ev["date"]][0] + lead
        for d in range(4):
            i = onset + d
            if i < n:
                score[i] += ev["impact"] * np.exp(-decay * d)
    return score


def inject_events(series: np.ndarray, events: pd.DataFrame,
                  date_index: pd.Series, decay: float | None = None) -> np.ndarray:
    """Замкнутая валидация: умножает ряд на exp(impact * weight) в окне события."""
    decay = decay if decay is not None else get("news.decay", 0.7)
    out = series.copy()
    for _, ev in events.iterrows():
        if ev["kind"] == "noise":
            continue
        onset = date_index.index[date_index >= ev["date"]][0] + ev["true_lead"]
        for d in range(4):
            i = onset + d
            if i < len(out):
                out[i] *= np.exp(ev["impact"] * np.exp(-decay * d))
    return out


def norm_ccf(x: np.ndarray, y: np.ndarray, max_lag: int = 6) -> pd.Series:
    """Нормированная кросс-корреляция: ccf[k] = corr(x[t], y[t+k]), |ccf| <= 1."""
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    n = len(x)
    out = {}
    for k in range(-max_lag, max_lag + 1):
        a, b = (x[:n - k], y[k:]) if k >= 0 else (x[-k:], y[:n + k])
        out[k] = float(np.mean(a * b))
    return pd.Series(out)


def estimate_lead(events: pd.DataFrame, series: np.ndarray,
                  date_index: pd.Series, decay: float | None = None) -> tuple[int, pd.DataFrame]:
    """Оценка лага опережения новостей перебором кандидатов из config.

    Returns: (L_hat, таблица candidate_lead -> corr).
    """
    grid = get("news.lead_grid", [0, 1, 2, 3, 4])
    rows = []
    for L in grid:
        sc = build_score(events, date_index, lead_override=L, decay=decay)
        rows.append([L, round(float(np.corrcoef(sc, series)[0, 1]), 3)])
    tab = pd.DataFrame(rows, columns=["candidate_lead", "corr"])
    return int(tab.loc[tab["corr"].idxmax(), "candidate_lead"]), tab


def seasonal_residual(log_series: np.ndarray, nat_log: np.ndarray) -> pd.Series:
    """Сезонно-скорректированный остаток: YoY ряда минус YoY национального индекса."""
    n = len(log_series)
    res = pd.Series(np.nan, index=range(n))
    for t in range(12, n):
        res[t] = (log_series[t] - log_series[t - 12]) - (nat_log[t] - nat_log[t - 12])
    return res


def event_study(events: pd.DataFrame, resid: pd.Series, date_index: pd.Series,
                car_window: int | None = None,
                quiet_thr: float = 0.05) -> pd.DataFrame:
    """CAR за car_window месяцев после onset; проверка знака/шума.

    Шумовые события, чьё окно пересекает окна impact-событий, помечаются
    'окно шока' и исключаются из оценки точности (флаг там корректен по смыслу).
    """
    car_window = car_window or get("news.event_study.car_window", 3)
    imp_win = [
        (date_index.index[date_index >= e.date][0] + e.true_lead - 1,
         date_index.index[date_index >= e.date][0] + e.true_lead + car_window)
        for _, e in events[events.kind != "noise"].iterrows()
    ]
    rows = []
    for _, ev in events.iterrows():
        o = date_index.index[date_index >= ev["date"]][0] + ev["true_lead"]
        if o < 12 or o + car_window > len(resid):
            rows.append([ev["date"].strftime("%Y-%m"), ev["kind"], round(ev["impact"], 2), np.nan, "вне окна"])
            continue
        if ev["kind"] == "noise" and any(a <= o <= b for a, b in imp_win):
            rows.append([ev["date"].strftime("%Y-%m"), ev["kind"], round(ev["impact"], 2), np.nan, "окно шока"])
            continue
        car = float(resid.iloc[o:o + car_window].mean())
        ok = (abs(car) < quiet_thr) if ev["kind"] == "noise" \
            else (np.sign(car) == np.sign(ev["impact"]))
        rows.append([ev["date"].strftime("%Y-%m"), ev["kind"], round(ev["impact"], 2), round(car, 3), ok])
    return pd.DataFrame(rows, columns=["event", "type", "impact", "CAR", "sign_match"])
