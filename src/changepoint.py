"""Обнаружение точек структурных изменений: оффлайн-методы и онлайн-слой.

Оффлайн: PELT (l2/rbf), Binseg, Window, классический CUSUM.
Онлайн: CUSUM на остатках прогнозной модели (раннее предупреждение).
Выбор метода обоснован симуляционным бенчмарком на ground truth.
"""
import numpy as np
import pandas as pd
import ruptures as rpt

from .config import get


# ---------------------------------------------------------------- оффлайн
def bic_pen(s: np.ndarray) -> float:
    """BIC-штраф за точку изменения для PELT (l2)."""
    d = np.diff(s)
    v = d.var() if d.var() > 0 else 1e-6
    return max(np.log(len(s)) * v, 1e-6)


def cusum_cp(s: np.ndarray, k: float = 0.5, h: float = 3.5) -> list[int]:
    """Классический двусторонний CUSUM на стандартизованном ряде."""
    z = (s - s.mean()) / (s.std() + 1e-9)
    sp = sn = 0.0
    cps = []
    for i, v in enumerate(z):
        sp = max(0.0, sp + v - k)
        sn = max(0.0, sn - v - k)
        if sp > h or sn > h:
            cps.append(i)
            sp = sn = 0.0
    return cps


def get_methods() -> dict:
    """Словарь оффлайн-детекторов: имя -> функция(ряд log) -> список точек."""
    cfg = get("changepoint.offline_best", {})
    width = cfg.get("width", 6)
    nb = cfg.get("n_bkps", 3)
    return {
        "PELT_l2": lambda s: rpt.Pelt(model="l2", min_size=3).fit(s).predict(pen=bic_pen(s))[:-1],
        "PELT_rbf": lambda s: rpt.Pelt(model="rbf", min_size=3).fit(s).predict(pen=1.0)[:-1],
        "Binseg": lambda s: rpt.Binseg(model="l2", min_size=3).fit(s).predict(n_bkps=nb)[:-1],
        "Window": lambda s: rpt.Window(width=width, model="l2").fit(s).predict(n_bkps=nb)[:-1],
        "CUSUM": lambda s: cusum_cp(s),
    }


# ---------------------------------------------------------------- бенчмарк
def inject_shock(s: np.ndarray, t0: int, kind: str, mag: float) -> np.ndarray:
    """Внедряет шок известного типа: level (сдвиг), spike (всплеск), trend (излом)."""
    s = s.copy()
    n = len(s)
    if kind == "level":
        s[t0:] += mag
    elif kind == "spike":
        s[t0:min(t0 + 2, n)] += mag
    else:
        s[t0:] += mag * np.arange(n - t0) / 6.0
    return s


def run_benchmark(LOG: np.ndarray, col_idx: list[int],
                  n_trials: int = 300, seed: int = 7) -> pd.DataFrame:
    """Симуляционный бенчмарк: внедряет известные шоки и измеряет детекцию.

    Returns: DataFrame[method, kind, recall, precision, n_cp, delay].
    """
    rng = np.random.default_rng(seed)
    methods = get_methods()
    rows = []
    for _ in range(n_trials):
        j = int(rng.choice(col_idx))
        s0 = LOG[j].copy()
        t0 = int(rng.integers(8, 19))
        kind = str(rng.choice(["level", "spike", "trend"]))
        mag = float(rng.uniform(*get("changepoint.benchmark.magnitude_range", [0.25, 0.6]))) \
            * float(rng.choice([1, -1]))
        s = inject_shock(s0, t0, kind, mag)
        for name, fn in methods.items():
            try:
                cps = fn(s)
            except Exception:
                cps = []
            tp = [d for d in cps if t0 - 1 <= d <= t0 + 3]
            rows.append({
                "method": name, "kind": kind,
                "recall": 1 if tp else 0,
                "precision": len(tp) / len(cps) if cps else 0.0,
                "n_cp": len(cps),
                "delay": (min(tp) - t0) if tp else np.nan,
            })
    return pd.DataFrame(rows)


def summarize(bench: pd.DataFrame) -> pd.DataFrame:
    """Сводка бенчмарка: recall/precision/F1/задержка по методам."""
    g = bench.groupby("method").agg(
        recall=("recall", "mean"), precision=("precision", "mean"),
        n_cp=("n_cp", "mean"), delay=("delay", "mean"))
    g["F1"] = 2 * g.recall * g.precision / (g.recall + g.precision)
    return g.round(3).sort_values("F1", ascending=False)


def agreement_matrix(cps_by_method: dict) -> pd.DataFrame:
    """Матрица согласованности методов (нечёткое совпадение точек ±1 мес)."""
    names = list(cps_by_method)
    def fuzzy(A, B, tol=1):
        return sum(1 for a in A if any(abs(a - b) <= tol for b in B)) / max(len(A), 1)
    return pd.DataFrame(
        {n1: [np.mean([fuzzy(a, b) for a, b in zip(cps_by_method[n1], cps_by_method[n2])])
              for n2 in names] for n1 in names},
        index=names).round(2)


# ---------------------------------------------------------------- онлайн
def online_cusum(r: np.ndarray, k: float | None = None,
                 h_thr: float | None = None, init: int | None = None) -> list[int]:
    """Онлайн-CUSUM на остатках модели: индексы тревог.

    Параметры по умолчанию берутся из config.yaml (online_cusum).
    """
    cfg = get("online_cusum", {})
    k = k if k is not None else cfg.get("k", 0.3)
    h_thr = h_thr if h_thr is not None else cfg.get("h_thr", 2.5)
    init = init if init is not None else cfg.get("init_window", 6)
    if len(r) < init + 2:
        return []
    mu, sd = r[:init].mean(), r[:init].std() + 1e-9
    z = (r - mu) / sd
    sp = sn = 0.0
    alarms = []
    for i, v in enumerate(z):
        sp = max(0.0, sp + v - k)
        sn = max(0.0, sn - v - k)
        if sp > h_thr or sn > h_thr:
            alarms.append(i)
            sp = sn = 0.0
    return alarms


def evaluate_online(alarms_by_mo: dict, window_cps_by_mo: dict,
                    t_min: int = 13, t_max: int = 23,
                    confirm_width: int = 6) -> tuple[pd.Series, pd.Series]:
    """Задержка детекции и выигрыш реального времени против оффлайн-подтверждения.

    Оффлайн-метод с окном W подтверждает точку лишь спустя ~W месяцев после неё;
    онлайн-тревога приходит через delay месяцев → gain = W - delay.
    """
    delays, gains = [], []
    for mo, cps in window_cps_by_mo.items():
        al = alarms_by_mo.get(mo, [])
        for c in cps:
            if not (t_min <= c <= t_max):
                continue
            hit = [a for a in al if c <= a <= c + 3]
            if hit:
                delays.append(min(hit) - c)
                gains.append(confirm_width - (min(hit) - c))
    return pd.Series(delays), pd.Series(gains)
