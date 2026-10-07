# Прогнозирование потребительских расходов муниципалитетов РФ и раннее обнаружение структурных шоков

**Онлайн-конкурс СберИндекса 2026 · номинация «Прогнозирование»**
**Команда:** FutureMap

## Навигация по проекту

### Документы для конкурса
| Документ | Ссылка |
|---|---|
| Методологический отчет в ПДФ | [`report/report.pdf`](report/report.pdf)
| Презентация| [`report/presentation.pdf`](report/presentation.pdf)
| Инструкция по запуску | [run_all.sh](run_all.sh) - запуск одной командой
| Конфигурационные файлы |  [config.yaml](config.yaml) |

### Код и воспроизводимость
| Файл | Содержание |
|---|---|
| [`notebooks/01_eda.py`](notebooks/01_eda.py) | Разведочный анализ: распределения, сезонность, STL, ADF |
| [`notebooks/02_forecast.py`](notebooks/02_forecast.py) | Prophet vs LightGBM vs Chronos-T5, тест Уилкоксона, ablation, robustness |
| [`notebooks/03_changepoints.py`](notebooks/03_changepoints.py) | 5 детекторов точек изменения, бенчмарк 300 шоков, онлайн-CUSUM |
| [`notebooks/04_news_alignment.py`](notebooks/04_news_alignment.py) | Новости: news_score, lead-grid, event study, кейс Анадыря |
| [`notebooks/05_final_experiments.py`](notebooks/05_final_experiments.py) | Финальные эксперименты |
| [`config.yaml`](config.yaml) | Все гиперпараметры: модель, детекторы, CUSUM, новости |
| [`requirements.txt`](requirements.txt) | Зависимости Python |
| [`run_all.sh`](run_all.sh) | Запуск проекта |
| [`data/README.md`](data/README.md) | Источники данных: СберИндекс, Росстат БДПМО, ЦБ РФ, новости |

### Результаты (таблицы)
| Файл | Описание |
|---|---|
| [`results/final_model_comparison_full.csv`](results/final_model_comparison_full.csv) | MAE/WAPE/R²: Prophet vs LightGBM vs Chronos, h = 1/3/6/12 |
| [`results/wilcoxon_test.csv`](results/wilcoxon_test.csv) | Статистическая значимость превосходства LightGBM |
| [`results/ablation_features.csv`](results/ablation_features.csv) | Вклад групп признаков |
| [`results/robustness_hyperparams.csv`](results/robustness_hyperparams.csv) | Устойчивость к гиперпараметрам |
| [`results/changepoint_summary.csv`](results/changepoint_summary.csv) | Бенчмарк 5 детекторов (Precision/Recall/F1) |
| [`results/changepoint_benchmark.csv`](results/changepoint_benchmark.csv) | Построчные результаты 300 синтетических шоков |
| [`results/online_cusum_metrics.csv`](results/online_cusum_metrics.csv) | Задержка раннего обнаружения и выигрыш времени |
| [`results/ablation_detectors.csv`](results/ablation_detectors.csv) | Сравнение конфигураций ансамбля детекторов |
| [`results/events_real.csv`](results/events_real.csv) | 5 событий Анадыря|
| [`results/event_study_real.csv`](results/event_study_real.csv) | Event study по событиям |
| [`results/ab_test_news_score_real.csv`](results/ab_test_news_score_real.csv) | A/B-тест интеграции news_score в модель |
| [`results/ab_test_bdmo.csv`](results/ab_test_bdmo.csv) | A/B-тест признаков Росстата (БДПМО) |
| [`results/worst_predictions.csv`](results/worst_predictions.csv) | Анализ ошибок: систематический паттерн января |

### Графики
18 графиков в [`figures/`](figures/) (EDA, сравнение моделей, бенчмарк детекторов, кейсы, слайдовые схемы). Подписи — в [`figures/CAPTIONS.md`](figures/CAPTIONS.md).

---

## Ключевые результаты

| Результат | Значение | Раздел отчета |
|---|---|---|
| LightGBM vs Prophet (MAE, h = 1…12) | **−30…−80%**, p < 0.001 (Уилкоксон) | §3.1 |
| LightGBM vs Chronos-T5 (zero-shot) | лучше в **1.9–3.7×** | §3.2 |
| Признаки Росстата (БДПМО), A/B-тест | +2.22% MAE — статика не помогает месячному прогнозу | §3.3 |
| Лучший детектор точек изменения | **Window, F1 = 0.592** | §4.1 |
| Раннее обнаружение (онлайн-CUSUM) | задержка **1.69 мес**, выигрыш **4.31 мес** | §4.2 |
| Новости: кейс Анадыря (МО 2344) | CCF = 0.66, схема согласования валидирована | §5 |

---

## Быстрый старт

```bash
git clone https://github.com/belashandrej-png/sberindex-mo-forecast-shocks.git
cd sberindex-mo-forecast-shocks
pip install -r requirements.txt
# данные конкурса положить в data/hackathonlicence/ (инструкция: data/README.md)
bash run_all.sh
