"""Загрузка и первичная агрегация данных СберИндекса.

Ключевое методологическое правило: таргет — строка категории
«Все категории»; суммирование отдельных категорий запрещено
(двойной счёт из-за скрытых мелких категорий).
"""
import numpy as np
import pandas as pd

from .config import get, resolve_data_dir


def load_consumption() -> pd.DataFrame:
    """Расходы МО: date (datetime), territory_id, category, value."""
    df = pd.read_parquet(resolve_data_dir() / get("data.files.consumption"))
    if df["date"].dtype == object:                      # строка '2023-01'
        df["date"] = pd.to_datetime(df["date"] + "-01")
    else:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_connection() -> pd.DataFrame:
    """Связи МО: territory_id_x, territory_id_y, distance, type."""
    return pd.read_parquet(resolve_data_dir() / get("data.files.connection"))


def load_market_access() -> pd.DataFrame:
    """Индекс доступности рынков: territory_id, market_access."""
    return pd.read_parquet(resolve_data_dir() / get("data.files.market_access"))


def prepare_panel(consumption: pd.DataFrame) -> dict:
    """Строит матрицу Y (месяцы x МО) по категории-таргету.

    Returns:
        dict: months (list[Timestamp]), T, mos (list[id]),
              pos (id->колоночный индекс), Yv (np.ndarray[T, N]).
    """
    target = get("data.target_category", "Все категории")
    tot = consumption[consumption["category"] == target][
        ["territory_id", "date", "value"]
    ]
    months = sorted(tot["date"].unique())
    cnt = tot.groupby("territory_id")["date"].count()
    full = sorted(cnt[cnt == len(months)].index)        # только полная история
    Y = tot[tot["territory_id"].isin(full)].pivot(
        index="date", columns="territory_id", values="value"
    ).sort_index()
    Y.index = range(len(months))
    return {
        "months": months,
        "T": len(months),
        "mos": list(Y.columns),
        "pos": {m: j for j, m in enumerate(Y.columns)},
        "Yv": Y.values.astype(float),
    }


def national_index(Yv: np.ndarray) -> np.ndarray:
    """Национальный ряд (сумма по МО) — прокси общего макрофактора."""
    return Yv.sum(axis=1)


def category_shares(consumption: pd.DataFrame, months: list, mos: list) -> dict:
    """Доли категорий на уровне МО, матрицы (T x N) для 'Продовольствие'/'Маркетплейсы'."""
    piv = consumption.pivot_table(
        index=["date", "territory_id"], columns="category",
        values="value", aggfunc="sum", fill_value=0,
    )
    tot = piv.sum(axis=1).replace(0, np.nan)
    idx = pd.MultiIndex.from_product(
        [sorted(consumption["date"].unique()), mos],
        names=["date", "territory_id"],
    )
    out = {}
    for c in get("features.category_shares_full",
                 ["Продовольствие", "Маркетплейсы"]):
        out[c] = (piv[c] / tot).reindex(idx).values.reshape(len(months), len(mos))
    return out


def static_features(connection: pd.DataFrame, market_access: pd.DataFrame,
                    mos: list) -> tuple[np.ndarray, np.ndarray]:
    """Пространственные признаки: min_dist и market_access (векторы по МО).

    degree не используется: сеть почти полная (средняя степень ~4600),
    признак вырожден (см. EDA, notebook 01).
    """
    mdist = pd.concat([
        connection.groupby("territory_id_x")["distance"].min(),
        connection.groupby("territory_id_y")["distance"].min(),
    ]).groupby(level=0).min()
    net = pd.DataFrame({"min_dist": mdist}).join(
        market_access.set_index("territory_id")
    ).reindex(mos)
    return net["min_dist"].values, net["market_access"].values


def stratified_sample(Yv: np.ndarray, mos: list,
                      n_per_decile: int = 30, seed: int = 42) -> list:
    """Стратифицированная выборка МО: по n из каждого дециля суммарных трат.

    Нужна для честного head-to-head сравнения моделей на одном наборе МО.
    """
    rng = np.random.default_rng(seed)
    ts = pd.Series(Yv.sum(axis=0), index=mos)
    dec = pd.qcut(ts.rank(method="first"), 10, labels=False)
    return list(np.concatenate([
        rng.choice(ts.index[dec == d], n_per_decile, replace=False)
        for d in range(10)
    ]))


def build_context() -> dict:
    """Собирает всё необходимое для признаков и моделей в один словарь."""
    cons = load_consumption()
    ctx = prepare_panel(cons)
    ctx["nat_v"] = national_index(ctx["Yv"])
    ctx["SHM"] = category_shares(cons, ctx["months"], ctx["mos"])
    ctx["minv"], ctx["mav"] = static_features(
        load_connection(), load_market_access(), ctx["mos"]
    )
    ctx["sample_mos"] = stratified_sample(
        ctx["Yv"], ctx["mos"],
        n_per_decile=get("backtest.n_per_decile", 30),
        seed=get("project.seed", 42),
    )
    return ctx
