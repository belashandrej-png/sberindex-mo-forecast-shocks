# Обнаружение точек структурных изменений:
import ruptures as rpt
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

LOG = np.log1p(Y)   # лог-ряды (стабилизация дисперсии)

# 5 методов детекции
def bic_pen(s):
    d = np.diff(s)
    v = d.var() if d.var() > 0 else 1e-6
    return max(np.log(len(s)) * v, 1e-6)

def cusum_cp(s, k=0.5, h=3.5):
    s = np.asarray(s, float)
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

METHODS = {
    'PELT_l2':  lambda s: rpt.Pelt(model='l2',  min_size=3).fit(s).predict(pen=bic_pen(s))[:-1],
    'PELT_rbf': lambda s: rpt.Pelt(model='rbf', min_size=3).fit(s).predict(pen=1.0)[:-1],
    'Binseg':   lambda s: rpt.Binseg(model='l2', min_size=3).fit(s).predict(n_bkps=3)[:-1],
    'Window':   lambda s: rpt.Window(width=6, model='l2').fit(s).predict(n_bkps=3)[:-1],
    'CUSUM':    lambda s: cusum_cp(s),
}

# 1. Симуляционный бенчмарк 300 испытаний
rng = np.random.default_rng(7)

def inject(s, t0, kind, mag):
    s = s.copy(); n = len(s)
    if kind == 'level':  s[t0:] += mag                      # сдвиг уровня
    elif kind == 'spike': s[t0:min(t0+2, n)] += mag         # всплеск
    else: s[t0:] += mag * np.arange(n - t0) / 6.0           # излом тренда
    return s

rows = []
for trial in range(300):
    mo  = rng.choice(list(sample_mos))
    s0  = LOG[mo].values.astype(float)
    t0  = int(rng.integers(8, 19))
    kind = rng.choice(['level', 'spike', 'trend'])
    mag  = rng.uniform(0.25, 0.6) * rng.choice([1, -1])
    s = inject(s0, t0, kind, mag)
    for name, fn in METHODS.items():
        t_st = time.time()
        try:    cps = fn(s)
        except Exception: cps = []
        tp = [d for d in cps if t0 - 1 <= d <= t0 + 3]      # окно допуска
        rows.append({
            'method': name, 'kind': kind,
            'recall': 1 if tp else 0,
            'precision': (len(tp) / len(cps)) if cps else 0.0,
            'n_cp': len(cps),
            'delay': (min(tp) - t0) if tp else np.nan,
            'time': time.time() - t_st,
        })

bench = pd.DataFrame(rows)
summary = bench.groupby('method').agg(
    recall=('recall', 'mean'), precision=('precision', 'mean'),
    n_cp=('n_cp', 'mean'), delay=('delay', 'mean'), time=('time', 'mean'))
summary['F1'] = 2 * summary.recall * summary.precision / (summary.recall + summary.precision)
summary = summary.round(3).sort_values('F1', ascending=False)
print("Бенчмарк методов (300 синтетических шоков в реальных рядах)")
print(summary)
best_method = summary.index[0]
print(f"\n Лучший метод по F1: {best_method}")

print("\n Recall по типам шоков")
print(bench.pivot_table(index='method', columns='kind', values='recall', aggfunc='mean').round(3))

# 2. Применение к реальным данным
store = {name: [] for name in METHODS}
for mo in sample_mos:
    s = LOG[mo].values.astype(float)
    for name, fn in METHODS.items():
        try:    store[name].append(fn(s))
        except Exception: store[name].append([])

print("\n Реальные данные (300 МО)")
for name in METHODS:
    cnts = [len(c) for c in store[name]]
    share = np.mean([c > 0 for c in cnts])
    print(f"  {name}: среднее число точек = {np.mean(cnts):.2f}, "
          f"доля МО с ≥1 шоком = {share:.2f}")

# Согласованность методов (нечеткое совпадение ±1 месяц)
def fuzzy(A, B, tol=1):
    return sum(1 for a in A if any(abs(a - b) <= tol for b in B)) / max(len(A), 1)
agree = pd.DataFrame(index=METHODS, columns=METHODS, dtype=float)
for n1 in METHODS:
    for n2 in METHODS:
        agree.loc[n1, n2] = np.mean([fuzzy(a, b) for a, b in zip(store[n1], store[n2])])
print("\nМатрица согласованности методов (доля совпадений):")
print(agree.round(2))

# 3. КЕЙС-СТАДИ: МО 2344
mo_case = 2344 if 2344 in Y.columns else sample_mos[0]
s = LOG[mo_case].values.astype(float)
cps = METHODS[best_method](s)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(range(len(Y[mo_case])), Y[mo_case].values, color='steelblue', lw=2, label='Факт (траты)')
for cp in cps:
    ax.axvline(cp, color='red', ls='--', lw=1.5)
    ax.annotate(months[cp].strftime('%Y-%m'), xy=(cp, Y[mo_case].iloc[cp]),
                xytext=(cp + 0.3, Y[mo_case].max() * 0.95),
                color='red', fontsize=10, fontweight='bold')
ax.set_title(f'Кейс: МО {mo_case} - точки структурных изменений ({best_method})',
             fontweight='bold', fontsize=13)
ax.set_xticks(range(len(months)))
ax.set_xticklabels([m.strftime('%y-%m') for m in months], rotation=45)
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('/content/case_study_shocks.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nДаты шоков МО {mo_case}: " + ", ".join(months[c].strftime('%Y-%m') for c in cps))

bench.to_csv('/content/changepoint_benchmark.csv', index=False)
summary.to_csv('/content/changepoint_summary.csv')
