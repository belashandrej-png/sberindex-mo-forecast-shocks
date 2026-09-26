# Визуализация распределений
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Настройки графиков
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Загрузка данных
consumption = pd.read_parquet('/content/hackathon/hackathonlicence/consumption.parquet')
market_access = pd.read_parquet('/content/hackathon/hackathonlicence/market_access.parquet')

# Преобразуем дату
if consumption['date'].dtype == object:
    consumption['date'] = pd.to_datetime(consumption['date'] + '-01')
else:
    consumption['date'] = pd.to_datetime(consumption['date'])

consumption['month'] = consumption['date'].dt.month
consumption['year']  = consumption['date'].dt.year

print(f"Всего записей: {consumption.shape[0]:,}")
print(f"Категорий: {consumption['category'].nunique()}")
print(f"Период: {consumption['date'].min().date()} — {consumption['date'].max().date()}")

# ГРАФИКИ
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
month_names = ['Янв','Фев','Мар','Апр','Май','Июн','Июл','Авг','Сен','Окт','Ноя','Дек']

# 1. Гистограмма распределения трат (в лог-шкале)
ax = axes[0, 0]
sample_data = consumption['value'].sample(min(50000, len(consumption)), random_state=42)
log_sample = np.log1p(sample_data)
sns.histplot(log_sample, bins=80, kde=True, ax=ax, color='steelblue')
ax.set_title('Распределение log(1 + траты)', fontweight='bold')
ax.set_xlabel('log(1 + value)')
ax.set_ylabel('Частота')
ax.axvline(log_sample.mean(), color='red', linestyle='--', label=f'Среднее: {log_sample.mean():.2f}')
ax.axvline(log_sample.median(), color='green', linestyle='--', label=f'Медиана: {log_sample.median():.2f}')
ax.legend(fontsize=9)

# 2. Box plot по категориям (лог-шкала)
ax = axes[0, 1]
top_categories = consumption.groupby('category')['value'].sum().nlargest(6).index
sample_cats = consumption[consumption['category'].isin(top_categories)].sample(
    min(15000, len(consumption[consumption['category'].isin(top_categories)])), random_state=42)
sns.boxplot(data=sample_cats, x='category', y='value', ax=ax, palette='Set2')
ax.set_title('Box Plot: Траты по категориям', fontweight='bold')
ax.set_xlabel('Категория')
ax.set_ylabel('Сумма трат')
ax.set_yscale('log')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

# 3. Violin plot по категориям
ax = axes[0, 2]
sample_cats_log = sample_cats.copy()
sample_cats_log['log_value'] = np.log1p(sample_cats_log['value'])
sns.violinplot(data=sample_cats_log, x='category', y='log_value', ax=ax, palette='muted', inner='quartile')
ax.set_title('Violin Plot: log(1 + траты)', fontweight='bold')
ax.set_xlabel('Категория')
ax.set_ylabel('log(1 + value)')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

# 4. Временные ряды ТОП-5 МО
ax = axes[1, 0]
top5_mos = consumption.groupby('territory_id')['value'].sum().nlargest(5).index
for mo in top5_mos:
    mo_data = consumption[consumption['territory_id'] == mo].groupby('date')['value'].sum()
    ax.plot(mo_data.index, mo_data.values, label=f'МО {mo}', linewidth=1.5)
ax.set_title('Временные ряды: ТОП-5 МО', fontweight='bold')
ax.set_xlabel('Дата')
ax.set_ylabel('Сумма трат')
ax.legend(fontsize=8, loc='upper left')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# 5. Сезонность по месяцам
ax = axes[1, 1]
monthly_avg = consumption.groupby('month')['value'].mean()
ax.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, color='coral')
ax.set_title('Сезонность: средние траты по месяцам', fontweight='bold')
ax.set_xlabel('Месяц')
ax.set_ylabel('Средняя сумма трат')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names, fontsize=9)
ax.grid(True, alpha=0.3)

# 6. Распределение market_access
ax = axes[1, 2]
sns.histplot(market_access['market_access'], bins=40, kde=True, ax=ax, color='purple')
ax.set_title('Распределение Market Access Index', fontweight='bold')
ax.set_xlabel('Market Access Index')
ax.set_ylabel('Количество МО')
ax.axvline(market_access['market_access'].mean(), color='red', linestyle='--',
           label=f'Среднее: {market_access["market_access"].mean():.1f}')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('/content/eda_distributions.png', dpi=150, bbox_inches='tight')
print("\nСохранено: eda_distributions.png")
plt.show()

# Сезонный паттерн по годам
fig, ax = plt.subplots(figsize=(14, 6))
yearly_data = consumption.groupby(['year', 'month'])['value'].mean().reset_index()
for year in sorted(consumption['year'].unique()):
    year_slice = yearly_data[yearly_data['year'] == year]
    ax.plot(year_slice['month'], year_slice['value'], marker='o', linewidth=2, label=str(int(year)))
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names, fontsize=10)
ax.set_title('Сезонный паттерн по годам', fontweight='bold', fontsize=13)
ax.set_xlabel('Месяц')
ax.set_ylabel('Средние траты по всем МО')
ax.legend(title='Год')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/content/seasonal_by_year.png', dpi=150, bbox_inches='tight')
print("Сохранено: seasonal_by_year.png")
plt.show()
