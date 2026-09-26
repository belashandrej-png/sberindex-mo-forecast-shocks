# БЛОК 1.6-FIX
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import wilcoxon

Yv = Y.values.astype(float)
nat_v = Y.sum(axis=1).values.astype(float)
T, N = Yv.shape
mos_arr = np.array(Y.columns.tolist())
pos = {mo: j for j, mo in enumerate(mos_arr)}

# Доли категорий на уровне МО (матрицы T×N)
dates_all = sorted(consumption['date'].unique())
piv_mo = consumption.pivot_table(index=['date','territory_id'], columns='category',
                                 values='value', aggfunc='sum', fill_value=0)
tot_mo = piv_mo.sum(axis=1).replace(0, np.nan)
idx = pd.MultiIndex.from_product([dates_all, mos_arr], names=['date','territory_id'])
SHM = {}
for cname in ['Продовольствие', 'Маркетплейсы']:
    s = (piv_mo[cname] / tot_mo).reindex(idx)
    SHM[cname] = s.values.reshape(len(dates_all), N)

min_dist_v = net['min_dist'].reindex(mos_arr).values
ma_v = net['market_access'].reindex(mos_arr).values

LAGS = [1, 2, 3, 6, 12, 13]

def featurize(t, j):
    """Признаки для прогноза месяца t по данным строго до t-1."""
    f = {}
    for lag in LAGS:
        if t - lag >= 0: f[f'y_lag{lag}'] = np.log1p(Yv[t-lag, j])
    for w in (3, 6, 12):
        if t - w >= 0:
            win = Yv[t-w:t, j]
            f[f'y_roll_mean_{w}'] = np.log1p(np.nanmean(win))
            f[f'y_roll_std_{w}'] = np.nanstd(win)
    if t >= 13:
        f['yoy_12'] = Yv[t-1, j] / max(Yv[t-12, j], 1) - 1   # БЕЗ утечки: только t-1
        f['yoy_13'] = Yv[t-1, j] / max(Yv[t-13, j], 1) - 1
    for lag in (1, 2, 4, 7, 13):                              # нац. индекс с лагом ≥1
        if t - lag >= 0: f[f'nat_lag{lag}'] = np.log1p(nat_v[t-lag])
    if t >= 13: f['nat_yoy'] = nat_v[t-1] / max(nat_v[t-13], 1) - 1
    if t >= 1:                                                # доли категорий на t-1
        f['share_food'] = SHM['Продовольствие'][t-1, j]
        f['share_mp']   = SHM['Маркетплейсы'][t-1, j]
    m = months[t].month                                       # календарь будущего известен
    f['month_sin'] = np.sin(2*np.pi*m/12); f['month_cos'] = np.cos(2*np.pi*m/12)
    f['is_dec'] = int(m == 12); f['is_jan'] = int(m == 1)
    f['trend'] = t
    f['min_dist'] = min_dist_v[j]; f['market_access'] = ma_v[j]
    return f

HORIZONS = [1, 3, 6, 12]
FCOLS = None
preds_fix = []
models_fix = {}

for h in HORIZONS:
    C = T - 1 - h
    rows, ys = [], []
    for mo in sample_mos:
        j = pos[mo]
        for t in range(1, C + 1):
            rows.append(featurize(t, j)); ys.append(np.log1p(Yv[t, j]))
    tr = pd.DataFrame(rows); tr_y = np.array(ys)
    if FCOLS is None:
        FCOLS = list(tr.columns)
    tr = tr.reindex(columns=FCOLS)

    model = lgb.LGBMRegressor(objective='mae', n_estimators=600, learning_rate=0.05,
                              num_leaves=63, min_child_samples=20, subsample=0.8,
                              colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(tr, tr_y)
    models_fix[h] = model

    test_feats, test_meta = [], []
    for t in range(C + 1, T):
        for mo in sample_mos:
            test_feats.append(featurize(t, pos[mo])); test_meta.append((mo, t))
    X_test = pd.DataFrame(test_feats).reindex(columns=FCOLS)
    yp = np.expm1(model.predict(X_test))
    for (mo, t), p in zip(test_meta, yp):
        preds_fix.append({'mo': mo, 'h': h, 't': t,
                          'y_true': Yv[t, pos[mo]], 'y_pred': max(p, 0)})
    print(f"  Horizon {h}: train={len(tr)}, test={len(test_meta)}")

preds_fix = pd.DataFrame(preds_fix)
preds_fix.to_parquet('/content/preds_lgb_fix.parquet')

def compute_metrics(preds_df):
    rows = []
    for h in HORIZONS:
        d = preds_df[preds_df.h == h]
        err = np.abs(d.y_true - d.y_pred)
        r2 = 1 - ((d.y_true-d.y_pred)**2).sum() / ((d.y_true-d.y_true.mean())**2).sum()
        rows.append([h, err.mean(), err.sum()/d.y_true.sum(), r2])
    return pd.DataFrame(rows, columns=['horizon','MAE','WAPE','R2']).round(3)

m_fix = compute_metrics(preds_fix)
m_pr  = compute_metrics(preds_prophet)
print("\n сравнение")
print("PROPHET:");  print(m_pr)
print("LIGHTGBM-FIX:"); print(m_fix)
print("\nΔ_MAE_%:", ((m_pr.MAE.values - m_fix.MAE.values) / m_pr.MAE.values * 100).round(1))
