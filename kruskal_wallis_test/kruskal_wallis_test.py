import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

DATA_PATH = "data.csv"

VALUE_COLUMN = "value"        # столбец с числовыми значениями
GROUP_COLUMN = "group"        # столбец с метками групп (3 и более)

ALPHA = 0.05
POSTHOC_METHOD = "bonferroni" # поправка для теста Данна: "bonferroni", "holm", "fdr_bh" ...
SAVE_FIGURE = True
FIGURE_PATH = "kruskal_wallis_plot.png"
# ==================================================================


def load_data(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def run_test(df):
    """Запускает критерий Краскела — Уоллиса."""
    data = df[[VALUE_COLUMN, GROUP_COLUMN]].dropna()
    group_names = data[GROUP_COLUMN].unique()

    if len(group_names) < 3:
        print(f"Внимание: групп всего {len(group_names)}. "
              f"Для 2 групп лучше использовать критерий Манна — Уитни.")

    # формируем список выборок по группам
    samples = [data.loc[data[GROUP_COLUMN] == g, VALUE_COLUMN] for g in group_names]

    h_stat, p_value = stats.kruskal(*samples)

    print("=== Критерий Краскела — Уоллиса ===")
    print(f"Число групп: {len(group_names)}\n")
    for g, s in zip(group_names, samples):
        print(f"  {g}: n = {len(s)}, медиана = {s.median():.3f}")

    print(f"\nH-статистика: {h_stat:.3f}")
    print(f"p-значение:   {p_value:.4f}")

    if p_value < ALPHA:
        print(f"\nВывод: между группами есть значимые различия (p < {ALPHA}).")
        run_posthoc(data)
    else:
        print(f"\nВывод: значимых различий между группами нет (p >= {ALPHA}).")

    return data


def run_posthoc(data):
    """Пост-хок тест Данна для попарных сравнений (если доступен пакет)."""
    try:
        import scikit_posthocs as sp
    except ImportError:
        print("\n(Пост-хок пропущен: установите scikit-posthocs для теста Данна.)")
        return

    print(f"\n--- Пост-хок тест Данна (поправка: {POSTHOC_METHOD}) ---")
    print("Матрица p-значений попарных сравнений:")
    p_matrix = sp.posthoc_dunn(
        data, val_col=VALUE_COLUMN, group_col=GROUP_COLUMN,
        p_adjust=POSTHOC_METHOD,
    )
    print(p_matrix.round(4))

    print("\nЗначимые пары:")
    cols = p_matrix.columns
    found = False
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i < j and p_matrix.loc[a, b] < ALPHA:
                print(f"  {a} vs {b}: p = {p_matrix.loc[a, b]:.4f}")
                found = True
    if not found:
        print("  (после поправки значимых пар не осталось)")


def plot_groups(data, save=False, path="plot.png"):
    """Boxplot со «скрипкой» для всех групп."""
    plt.figure(figsize=(9, 6))
    sns.violinplot(data=data, x=GROUP_COLUMN, y=VALUE_COLUMN, inner=None,
                   color="lightgray", cut=0)
    sns.boxplot(data=data, x=GROUP_COLUMN, y=VALUE_COLUMN, width=0.25,
                boxprops={"zorder": 2})
    sns.stripplot(data=data, x=GROUP_COLUMN, y=VALUE_COLUMN, color="black",
                  alpha=0.4, size=3, jitter=True)

    plt.title("Сравнение групп (Краскел — Уоллис)")
    plt.tight_layout()
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\nКартинка сохранена: {path}")
    plt.show()


def main():
    df = load_data(DATA_PATH)
    data = run_test(df)
    plot_groups(data, save=SAVE_FIGURE, path=FIGURE_PATH)


if __name__ == "__main__":
    main()
