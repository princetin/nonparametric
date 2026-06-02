import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

DATA_PATH = "data.csv"

MODE = "long"                 # "long" или "wide"

# Для MODE = "long":
VALUE_COLUMN = "value"        # столбец с числовыми значениями
GROUP_COLUMN = "group"        # столбец с метками групп (ровно 2 уникальные)

# Для MODE = "wide":
COLUMN_A = "group_a"          # числовой столбец первой группы
COLUMN_B = "group_b"          # числовой столбец второй группы

ALPHA = 0.05
ALTERNATIVE = "two-sided"     # "two-sided", "less" или "greater"
SAVE_FIGURE = True
FIGURE_PATH = "mann_whitney_plot.png"


def load_data(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def get_two_groups(df):
    """Возвращает (выборка_1, выборка_2, имя_1, имя_2) в зависимости от MODE."""
    if MODE == "long":
        groups = df[GROUP_COLUMN].dropna().unique()
        if len(groups) != 2:
            raise ValueError(
                f"Ожидалось ровно 2 группы в '{GROUP_COLUMN}', найдено {len(groups)}: {groups}"
            )
        name1, name2 = groups
        s1 = df.loc[df[GROUP_COLUMN] == name1, VALUE_COLUMN].dropna()
        s2 = df.loc[df[GROUP_COLUMN] == name2, VALUE_COLUMN].dropna()
        return s1, s2, str(name1), str(name2)

    elif MODE == "wide":
        return df[COLUMN_A].dropna(), df[COLUMN_B].dropna(), COLUMN_A, COLUMN_B

    raise ValueError("MODE должен быть 'long' или 'wide'")


def rank_biserial_effect(u, n1, n2):
    """Размер эффекта (ранговая бисериальная корреляция), от -1 до 1."""
    return 1 - (2 * u) / (n1 * n2)


def run_test():
    df = load_data(DATA_PATH)
    s1, s2, name1, name2 = get_two_groups(df)

    u_stat, p_value = stats.mannwhitneyu(s1, s2, alternative=ALTERNATIVE)
    effect = rank_biserial_effect(u_stat, len(s1), len(s2))

    print("=== Критерий Манна — Уитни ===")
    print(f"Группа 1: {name1}  (n = {len(s1)}, медиана = {s1.median():.3f})")
    print(f"Группа 2: {name2}  (n = {len(s2)}, медиана = {s2.median():.3f})")
    print(f"\nU-статистика:        {u_stat:.3f}")
    print(f"p-значение:          {p_value:.4f}")
    print(f"Размер эффекта (rb): {effect:+.3f}")
    print(f"Альтернатива:        {ALTERNATIVE}")

    if p_value < ALPHA:
        print(f"\nВывод: различия статистически значимы (p < {ALPHA}).")
    else:
        print(f"\nВывод: значимых различий не обнаружено (p >= {ALPHA}).")

    return s1, s2, name1, name2


def plot_groups(s1, s2, name1, name2, save=False, path="plot.png"):
    """Boxplot со «скрипкой» для наглядного сравнения распределений."""
    plot_df = pd.DataFrame(
        {"value": list(s1) + list(s2),
         "group": [name1] * len(s1) + [name2] * len(s2)}
    )

    plt.figure(figsize=(7, 6))
    sns.violinplot(data=plot_df, x="group", y="value", inner=None,
                   color="lightgray", cut=0)
    sns.boxplot(data=plot_df, x="group", y="value", width=0.25,
                showcaps=True, boxprops={"zorder": 2})
    sns.stripplot(data=plot_df, x="group", y="value", color="black",
                  alpha=0.4, size=3, jitter=True)

    plt.title("Сравнение двух групп (Манн — Уитни)")
    plt.tight_layout()
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Картинка сохранена: {path}")
    plt.show()


def main():
    s1, s2, name1, name2 = run_test()
    plot_groups(s1, s2, name1, name2, save=SAVE_FIGURE, path=FIGURE_PATH)


if __name__ == "__main__":
    main()
