# Разведочный анализ даннных
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Загружаем данные
consumption = pd.read_parquet('/content/hackathon/hackathonlicence/consumption.parquet')
connection = pd.read_parquet('/content/hackathon/hackathonlicence/connection.parquet')
market_access = pd.read_parquet('/content/hackathon/hackathonlicence/market_access.parquet')

print(f"\n1. ОСНОВНЫЕ ХАРАКТЕРИСТИКИ ДАННЫХ")
print(f"   Consumption: {consumption.shape[0]:,} записей")
print(f"   Connection: {connection.shape[0]:,} связей")
print(f"   Market Access: {market_access.shape[0]:,} МО")

# Преобразуем дату
consumption['date'] = pd.to_datetime(consumption['date'] + '-01')
consumption['year'] = consumption['date'].dt.year
consumption['month'] = consumption['date'].dt.month

print(f"\n2. Временной диапозон")
print(f"   Начало: {consumption['date'].min()}")
print(f"   Конец: {consumption['date'].max()}")
print(f"   Количество месяцев: {consumption['date'].nunique()}")

print(f"\n3. Муниципальные Образования")
print(f"   Уникальных МО: {consumption['territory_id'].nunique()}")
print(f"   МО в market_access: {market_access['territory_id'].nunique()}")

print(f"\n4. Категория трат")
categories = consumption['category'].unique()
print(f"   Количество категорий: {len(categories)}")
for i, cat in enumerate(categories, 1):
    print(f"   {i}. {cat}")
