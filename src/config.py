"""Загрузка конфигурации проекта (config.yaml).

Все гиперпараметры, пути и протоколы хранятся в одном YAML-файле
в корне репозитория; модули не содержат «зашитых» чисел.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]   # корень репозитория
CONFIG_PATH = ROOT / "config.yaml"

_CACHE = {}


def load_config(path: str | Path | None = None, reload: bool = False) -> dict:
    """Читает YAML-конфиг и кэширует результат.

    Args:
        path: путь к конфигу; по умолчанию <repo>/config.yaml.
        reload: принудительно перечитать файл.

    Returns:
        dict с конфигурацией.
    """
    path = Path(path) if path else CONFIG_PATH
    key = str(path)
    if key not in _CACHE or reload:
        with open(path, "r", encoding="utf-8") as f:
            _CACHE[key] = yaml.safe_load(f)
    return _CACHE[key]


def get(dotkey: str, default=None):
    """Доступ по точке: get('model.learning_rate') -> 0.05."""
    node = load_config()
    for part in dotkey.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolve_data_dir() -> Path:
    """Ищет папку с данными: локально в репозитории или в Colab."""
    local = ROOT / get("data.path", "data/hackathonlicence")
    if local.exists():
        return local
    colab = Path("/content/hackathon/hackathonlicence")
    if colab.exists():
        return colab
    raise FileNotFoundError(
        "Данные не найдены. Распакуйте архив конкурса в data/hackathonlicence/ "
        "(инструкция: data/README.md)."
    )
