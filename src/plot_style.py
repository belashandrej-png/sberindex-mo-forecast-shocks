import matplotlib.pyplot as plt

# Семантика цветов: одинаковый смысл = одинаковый цвет во всех фигурах
PALETTE = {
    "fact":     "#4C72B0",   # фактические траты
    "forecast": "#DD8452",   # прогноз модели
    "baseline": "#7F7F7F",   # Prophet / сравнения
    "shock":    "#C0392B",   # оффлайн-точки изменений
    "alarm":    "#E67E22",   # онлайн-тревоги (ранние)
    "news":     "#55A868",   # новостные события
}


def apply_style() -> None:
    """Применяет rcParams: чистая сетка, без верхних/правых рамок, жирные заголовки."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def save(fig, name: str, dpi: int = 150) -> None:
    """Сохраняет фигуру в figures/<name>.png без лишних полей."""
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=dpi, bbox_inches="tight")
