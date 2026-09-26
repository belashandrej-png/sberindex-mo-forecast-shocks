"""Прогнозные модели: LightGBM (финальная), Prophet (бейзлайн), Chronos (бенчмарк)."""
import logging
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import get
from .features import featurize, feature_columns


def _lgbm_params() -> dict:
    """Гиперпараметры LightGBM из config.yaml."""
    cfg = get("model", {})
    return dict(
        objective=cfg.get("objective", "mae"),        # метрика конкурса = лосс
        n_estimators=cfg.get("n_estimators", 600),
        learning_rate=cfg.get("learning_rate", 0.05),
        num_leaves=cfg.get("num_leaves", 63),
        max_depth=cfg.get("max_depth", 8),
        min_child_samples=cfg.get("min_child_samples", 20),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
        reg_alpha=cfg.get("reg_alpha", 0.1),
        reg_lambda=cfg.get("reg_lambda", 1.0),
        random_state=get("project.seed", 42),
        n_jobs=-1, verbose=-1,
    )


def train_lightgbm(ctx: dict, horizons: list[int]) -> tuple[dict, pd.DataFrame]:
    """Обучает по одной модели LightGBM на горизонт (прямая схема).

    Протокол last-block: для горизонта h обучение на месяцах <= T-1-h,
    тест на последних h месяцах. Признаки — строго до t-1.

    Returns:
        (models: dict[h -> LGBMRegressor], preds: DataFrame[mo, h, t, y_true, y_pred])
    """
    Yv, T, pos = ctx["Yv"], ctx["T"], ctx["pos"]
    FC = feature_columns(ctx)
    models, preds = {}, []
    for h in horizons:
        C = T - 1 - h
        rows, ys = [], []
        for mo in ctx["sample_mos"]:
            j = pos[mo]
            for t in range(1, C + 1):
                rows.append(featurize(t, j, ctx))
                ys.append(np.log1p(Yv[t, j]))
        tr = pd.DataFrame(rows).reindex(columns=FC)
        model = lgb.LGBMRegressor(**_lgbm_params())
        model.fit(tr, np.array(ys))
        models[h] = model

        tf, tm = [], []
        for t in range(C + 1, T):
            for mo in ctx["sample_mos"]:
                tf.append(featurize(t, pos[mo], ctx))
                tm.append((mo, t))
        yp = np.expm1(model.predict(pd.DataFrame(tf).reindex(columns=FC)))
        for (mo, t), p in zip(tm, yp):
            preds.append({"mo": mo, "h": h, "t": t,
                          "y_true": Yv[t, pos[mo]], "y_pred": max(float(p), 0.0)})
    return models, pd.DataFrame(preds)


def prophet_baseline(ctx: dict, horizons: list[int]) -> pd.DataFrame:
    """Бейзлайн конкурса: Prophet с мультипликативной годовой сезонностью."""
    from prophet import Prophet
    logging.getLogger("prophet").setLevel(logging.ERROR)

    Yv, T, months, pos = ctx["Yv"], ctx["T"], ctx["months"], ctx["pos"]
    rec = []
    for mo in ctx["sample_mos"]:
        d = pd.DataFrame({"t": range(T), "y": Yv[:, pos[mo]]})
        d["ds"] = d["t"].map(lambda i: months[i])
        for h in horizons:
            C = T - 1 - h
            tr, te = d[d.t <= C], d[d.t > C]
            m = Prophet(growth="linear", n_changepoints=4,
                        yearly_seasonality=False, weekly_seasonality=False,
                        daily_seasonality=False, seasonality_mode="multiplicative")
            m.add_seasonality(name="yearly", period=365.25, fourier_order=3)
            m.fit(tr[["ds", "y"]])
            fc = m.predict(m.make_future_dataframe(periods=h, freq="MS"))
            for y, p in zip(te["y"].values, fc["yhat"].tail(h).values):
                rec.append({"mo": mo, "h": h, "t": 0, "y_true": y, "y_pred": max(p, 0.0)})
    return pd.DataFrame(rec)


def chronos_zero_shot(ctx: dict, horizons: list[int]) -> tuple[str, pd.DataFrame]:
    """Foundation model Chronos без дообучения (роль: независимый бенчмарк).

    Пробуется Chronos-Bolt, при недоступности — Chronos-T5 (обе — foundation).
    Возвращается медиана прогнозного распределения.
    """
    import torch
    from chronos import ChronosPipeline

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        pipe = ChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map=dev)
        name = "Chronos-Bolt-Small"
    except Exception:
        pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map=dev)
        name = "Chronos-T5-Small"

    Yv, T, pos = ctx["Yv"], ctx["T"], ctx["pos"]
    rec = []
    for h in horizons:
        C = T - 1 - h
        ctxs = [torch.tensor(Yv[:C + 1, pos[mo]], dtype=torch.float32)
                for mo in ctx["sample_mos"]]
        fc = np.asarray(pipe.predict(ctxs, prediction_length=h,
                                     num_samples=get("foundation_model.num_samples", 20),
                                     limit_prediction_length=False))
        q = np.median(fc, axis=1)
        for i, mo in enumerate(ctx["sample_mos"]):
            for s in range(h):
                rec.append({"mo": mo, "h": h, "t": C + 1 + s,
                            "y_true": Yv[C + 1 + s, pos[mo]],
                            "y_pred": max(float(q[i, s]), 0.0)})
    return name, pd.DataFrame(rec)


def ensemble(preds_a: pd.DataFrame, preds_b: pd.DataFrame,
             w: float = 0.5) -> pd.DataFrame:
    """Простое взвешенное усреднение предсказаний двух моделей."""
    m = preds_a.merge(preds_b, on=["mo", "h", "t"], suffixes=("_a", "_b"))
    out = m[["mo", "h", "t", "y_true_a"]].rename(columns={"y_true_a": "y_true"})
    out["y_pred"] = w * m["y_pred_a"] + (1 - w) * m["y_pred_b"]
    return out
