#!/usr/bin/env bash

# sberindex-mo-forecast-shocks / run_all.sh
# Запуск:  bash run_all.sh
# Требуется: pip install -r requirements.txt  и данные в data/hackathonlicence/

set -e          # прервать выполнение при любой ошибке
set -o pipefail

echo "  СберИндекс 2026 "

# 0. Проверка окружения
python -c "import pandas, numpy, lightgbm, ruptures" 2>/dev/null || {
  echo "ОШИБКА: зависимости не установлены."
  echo "Выполните: pip install -r requirements.txt"
  exit 1
}

# 1. Проверка данных
if [ ! -f "data/hackathonlicence/consumption.parquet" ]; then
  echo "ОШИБКА: данные конкурса не найдены!"
  echo "Скачайте архив и распакуйте в data/hackathonlicence/"
  echo "Инструкция: data/README.md"
  exit 1
fi
echo "Данные найдены. Начинаю прогон..."

# 2. Последовательный запуск ноутбуков
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

# 3. Итог
echo ""
echo "  Все ноутбуки выполнены успешно!"
echo "  Таблицы результатов: results/*.csv"
echo "  Графики:             figures/*.png"
