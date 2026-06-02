import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

DATA_PATH = "data.csv"


COLUMNS = ["Олимп", "Тест", "Олимп", "Тест"]

ALPHA = 0.05
SAVE_FIGURE = True
FIGURE_PATH = "correlation_heatmaps.png"
# ==================================================================


def load_data(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def correlation_with_pvalues(df, method):
    cols = df.columns
    corr = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    pvals = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)

    func = stats.spearmanr if method == "spearman" else stats.kendalltau

    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i < j:
                pair = df[[a, b]].dropna()
                r, p = func(pair[a], pair[b])
                corr.loc[a, b] = corr.loc[b, a] = r
                pvals.loc[a, b] = pvals.loc[b, a] = p
    return corr, pvals


def print_summary(corr, pvals, method_name, alpha):
    print(f"\n=== {method_name} ===")
    cols = corr.columns
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i < j:
                r = corr.loc[a, b]
                p = pvals.loc[a, b]
                mark = "значимо" if p < alpha else "не значимо"
                print(f"  {a} ~ {b}: r = {r:+.3f}, p = {p:.4f}  ({mark})")


def plot_heatmaps(corr_s, corr_k, save=False, path="heatmaps.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, corr, title in zip(
        axes,
        [corr_s, corr_k],
        ["Корреляция Спирмана (ρ)", "Корреляция Кендалла (τ)"],
    ):
        sns.heatmap(
            corr,
            annot=True,          # подписать значения в ячейках
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1, vmax=1,     # фиксированная шкала для сопоставимости
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title(title, fontsize=13)

    plt.tight_layout()
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\nКартинка сохранена: {path}")
    plt.show()


def main():
    df = load_data(DATA_PATH)
    data = df[COLUMNS]

    corr_s, pvals_s = correlation_with_pvalues(data, "spearman")
    corr_k, pvals_k = correlation_with_pvalues(data, "kendall")

    print_summary(corr_s, pvals_s, "Спирман", ALPHA)
    print_summary(corr_k, pvals_k, "Кендалл", ALPHA)

    plot_heatmaps(corr_s, corr_k, save=SAVE_FIGURE, path=FIGURE_PATH)


if __name__ == "__main__":
    main()
