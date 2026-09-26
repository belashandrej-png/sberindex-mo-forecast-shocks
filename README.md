# sberindex-mo-forecast-shocks
Прогнозирование потребительских расходов МО РФ и раннее обнаружение структурных шоков.
Онлайн-конкурс СберИндекса 2026, номинация «Прогнозирование».

## Результаты
- LightGBM превосходит бейзлайн Prophet на 30-80% MAE (p < 0.001).
- Двухслойная детекция шоков (Window + онлайн-CUSUM).
- Интеграция новостей через event study (CAR).

**Требования:** Python 3.10+, ~4 ГБ RAM (CPU достаточно; GPU опционален для Chronos).

### Шаг 1. Зависимости
```bash
pip install -r requirements.txt

### Шаг 2. Данные
Скачайте архив конкурса и распакуйте в data/hackathonlicence/ — должны появиться
consumption.parquet, connection.parquet, market_access.parquet.
Инструкция и лицензия: data/README.md.

### Шаг 3. Запуск проекта
bash run_all.sh

### Шаг 4. Или вручную по шагам
jupyter notebook notebooks/01_eda.ipynb            # EDA, ~2 мин
jupyter notebook notebooks/02_forecast.ipynb       # Prophet / LightGBM / Chronos, ~15 мин
jupyter notebook notebooks/03_changepoints.ipynb   # детекция шоков, ~5 мин
jupyter notebook notebooks/04_news_alignment.ipynb # согласование новостей, ~1 мин

Где смотреть результат
Таблицы метрик: results/ (итог — final_model_comparison.csv)
Графики: figures/ (подписи-выводы: figures/CAPTIONS.md)
Отчет и презентация: report/
Воспроизводимость: seed=42 и все гиперпараметры — в config.yaml;
повторный запуск даёт побитово те же таблицы.


### 2. run_all.sh — создайте в корне (Add file → Create new file), содержимое:

```bash
#!/usr/bin/env bash
# Полный прогон пайплайна: bash run_all.sh
set -e
set -o pipefail

echo "  СберИндекс 2026: полный прогон пайплайна анализа"

python -c "import pandas, numpy, lightgbm, ruptures" 2>/dev/null || {
  echo "ОШИБКА: зависимости не установлены. Выполните: pip install -r requirements.txt"
  exit 1
}

if [ ! -f "data/hackathonlicence/consumption.parquet" ]; then
  echo "ОШИБКА: данные не найдены. Распакуйте архив конкурса в data/hackathonlicence/"
  echo "Инструкция: data/README.md"
  exit 1
fi

NOTEBOOKS=(
  "notebooks/01_eda.ipynb"
  "notebooks/02_forecast.ipynb"
  "notebooks/03_changepoints.ipynb"
  "notebooks/04_news_alignment.ipynb"
)

for nb in "${NOTEBOOKS[@]}"; do
  echo ""
  echo ">>> Выполняю $nb ..."
  jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=3600 "$nb"
  echo ">>> Готово: $nb"
done

echo ""
echo "  Все ноутбуки выполнены успешно!"
echo "  Таблицы: results/*.csv | Графики: figures/*.png"
