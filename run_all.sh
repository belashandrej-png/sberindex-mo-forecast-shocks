#!/usr/bin/env bash
# Полный прогон пайплайна: bash run_all.sh
set -e
set -o pipefail

echo "  СберИндекс 2026: пайплайна анализа"

# Проверка зависимостей
python -c "import pandas, numpy, lightgbm, ruptures" 2>/dev/null || {
  echo "ОШИБКА: зависимости не установлены. Выполните: pip install -r requirements.txt"
  exit 1
}

# Проверка данных
if [ ! -f "data/hackathonlicence/consumption.parquet" ]; then
  echo "ОШИБКА: данные не найдены. Распакуйте архив конкурса в data/hackathonlicence/"
  echo "Инструкция: data/README.md"
  exit 1
fi

# ИСПРАВЛЕНО: запуск .py файлов
SCRIPTS=(
  "notebooks/01_eda.py"
  "notebooks/02_forecast.py"
  "notebooks/03_changepoints.py"
  "notebooks/04_news_alignment.py"
)

for s in "${SCRIPTS[@]}"; do
  echo ""
  echo ">>> Выполняю $s ..."
  python "$s"
  echo ">>> Готово: $s"
done

echo ""
echo "  Все скрипты выполнены успешно!"
echo "  Таблицы: results/*.csv | Графики: figures/*.png"
