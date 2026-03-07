"""
Step 4: Compare NLP Matrices with Human Matrix
================================================
For each of the 28 NLP similarity matrices (24 BOW/TF-IDF + 4 WordNet),
extract the upper triangle (excluding diagonal) and compute the Pearson
correlation with the corresponding triangle of the human similarity matrix.

Inputs:
  step1/human_similarity_matrix.csv
  step2/cfg*.csv   (24 files)
  step3/wordnet_*.csv (4 files)

Output:
  step4/comparison_results.csv   — ranked table of Pearson r values
  step4/comparison_bar_chart.png — bar chart of all correlations
"""

import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import pearsonr

matplotlib.use("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = SCRIPT_DIR

HUMAN_CSV = os.path.join(ROOT_DIR, "step1", "csv", "human_similarity_matrix.csv")
BOW_TFIDF_DIR = os.path.join(ROOT_DIR, "step2", "csv")
WORDNET_DIR = os.path.join(ROOT_DIR, "step3", "csv")


def load_human_matrix(csv_path):
    human_df = pd.read_csv(csv_path, index_col=0)
    return human_df.values.astype(float)


def upper_triangle(mat, k=1):
    return mat[np.triu_indices(mat.shape[0], k=k)]


def load_nlp_files():
    nlp_files = sorted(glob.glob(os.path.join(BOW_TFIDF_DIR, "*.csv"))) + \
                sorted(glob.glob(os.path.join(WORDNET_DIR, "*.csv")))
    return nlp_files


def compute_correlations(human_tri, nlp_files, expected_shape):
    n = expected_shape[0]
    triu_idx = np.triu_indices(n, k=1)
    records = []
    for fpath in nlp_files:
        name = os.path.splitext(os.path.basename(fpath))[0]
        try:
            df = pd.read_csv(fpath, index_col=0)
            mat = df.values.astype(float)
            if mat.shape != expected_shape:
                raise ValueError(f"shape {mat.shape} != {expected_shape}")
            nlp_tri = mat[triu_idx]
            r, p = pearsonr(human_tri, nlp_tri)
            source = "WordNet" if "wordnet" in name else "BOW/TF-IDF"
            technique = _get_technique(name)
            records.append({"Matrix": name, "Source": source, "Technique": technique,
                            "Pearson_r": round(r, 6), "p_value": p})
            print(f"  {name:55s}  r = {r:+.4f}  p = {p:.4e}")
        except Exception as e:
            print(f"  [WARN] {name}: {e}")
    return records


def _get_technique(name: str) -> str:
    if "wordnet" in name:
        return "WordNet"
    return "BOW" if name.endswith("_BOW") else "TF-IDF"


_TECH_COLOR = {"BOW": "steelblue", "TF-IDF": "cornflowerblue", "WordNet": "darkorange"}
_TECH_COLOR_MAX = {"BOW": "#b30000", "TF-IDF": "#b30000", "WordNet": "#7a3800"}


def build_results_df(records):
    df = pd.DataFrame(records)
    df["abs_r"] = df["Pearson_r"].abs()
    df = df.sort_values("abs_r", ascending=False).drop(columns="abs_r")
    df.index = range(1, len(df) + 1)
    return df


def save_results_csv(results_df, out_dir):
    csv_out = os.path.join(out_dir, "csv", "comparison_results.csv")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    results_df.to_csv(csv_out)
    return csv_out


def save_bar_chart(results_df, out_dir):
    max_rows = set(results_df.groupby("Technique")["Pearson_r"].idxmax())
    chart_df = results_df.sort_values("Pearson_r").reset_index(drop=False)

    bar_colors, edge_colors, linewidths, is_max_list = [], [], [], []
    for _, row in chart_df.iterrows():
        tech   = row["Technique"]
        is_max = row["index"] in max_rows
        is_max_list.append(is_max)
        if is_max:
            bar_colors.append(_TECH_COLOR_MAX[tech])
            edge_colors.append("black")
            linewidths.append(1.5)
        else:
            bar_colors.append(_TECH_COLOR[tech])
            edge_colors.append("white")
            linewidths.append(0.5)

    fig_h = max(10, len(chart_df) * 0.35)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    bars = ax.barh(
        range(len(chart_df)), chart_df["Pearson_r"],
        color=bar_colors, edgecolor=edge_colors, linewidth=linewidths, height=0.8,
    )

    yticklabels = [
        f"\u2605 {row['Matrix']}" if row["index"] in max_rows else row["Matrix"]
        for _, row in chart_df.iterrows()
    ]
    ax.set_yticks(range(len(chart_df)))
    ax.set_yticklabels(yticklabels, fontsize=7)
    for tick, is_max in zip(ax.get_yticklabels(), is_max_list):
        if is_max:
            tick.set_fontweight("bold")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r  (vs. Human similarity matrix)", fontsize=11)
    ax.set_title(
        "NLP vs. Human Similarity: Pearson Correlation\n"
        "\u2605 Bold / dark bar = maximum per technique group",
        fontsize=12,
    )
    ax.set_xlim(0, 0.7)

    for bar, (_, row) in zip(bars, chart_df.iterrows()):
        is_max = row["index"] in max_rows
        ax.text(
            row["Pearson_r"] + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{row['Pearson_r']:.4f}", va="center", ha="left",
            fontsize=7, fontweight="bold" if is_max else "normal",
        )

    ax.legend(
        handles=[
            Patch(facecolor="steelblue",      label="BOW"),
            Patch(facecolor="cornflowerblue", label="TF-IDF"),
            Patch(facecolor="darkorange",     label="WordNet (GLAO)"),
            Patch(facecolor="#b30000",        label="Max per technique \u2605",
                  edgecolor="black", linewidth=1.2),
        ],
        loc="lower right", fontsize=8,
    )
    plt.tight_layout()
    png_out = os.path.join(out_dir, "png", "comparison_bar_chart.png")
    os.makedirs(os.path.dirname(png_out), exist_ok=True)
    plt.savefig(png_out, dpi=130)
    plt.close(fig)
    return png_out


def save_table_png(results_df, out_dir):
    """Save a 3-panel table PNG (BOW / TF-IDF / WordNet); max row highlighted in gold."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 10))
    fig.suptitle(
        "Pearson Correlation: NLP Matrices vs. Human Similarity Matrix\n"
        "(bold = maximum per technique)",
        fontsize=13, fontweight="bold",
    )
    for ax, tech in zip(axes, ["BOW", "TF-IDF", "WordNet"]):
        sub = results_df[results_df["Technique"] == tech].sort_values(
            "Pearson_r", ascending=False
        ).reset_index(drop=True)
        max_r = sub["Pearson_r"].max()
        cell_text = [
            [row["Matrix"], f"{row['Pearson_r']:.4f}", f"{row['p_value']:.3e}"]
            for _, row in sub.iterrows()
        ]
        tbl = ax.table(
            cellText=cell_text,
            colLabels=["Configuration", "Pearson r", "p-value"],
            cellLoc="center", loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.auto_set_column_width([0, 1, 2])
        for col in range(3):
            tbl[(0, col)].set_facecolor("#2c3e50")
            tbl[(0, col)].set_text_props(color="white", fontweight="bold")
        for row_i, (_, row) in enumerate(sub.iterrows(), start=1):
            if row["Pearson_r"] == max_r:
                for col in range(3):
                    tbl[(row_i, col)].set_facecolor("#ffd700")
                    tbl[(row_i, col)].set_text_props(fontweight="bold")
            elif row_i % 2 == 0:
                for col in range(3):
                    tbl[(row_i, col)].set_facecolor("#f0f0f0")
        ax.set_title(tech, fontsize=11, fontweight="bold", pad=10)
        ax.axis("off")
    plt.tight_layout()
    table_png = os.path.join(out_dir, "png", "comparison_table.png")
    os.makedirs(os.path.dirname(table_png), exist_ok=True)
    plt.savefig(table_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return table_png


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading human similarity matrix …")
    human_mat = load_human_matrix(HUMAN_CSV)
    n = human_mat.shape[0]
    human_tri = upper_triangle(human_mat)
    print(f"  Shape: {n}×{n}, upper-triangle elements: {len(human_tri)}")

    nlp_files = load_nlp_files()
    print(f"Found {len(nlp_files)} NLP similarity matrices to compare.")

    records = compute_correlations(human_tri, nlp_files, (n, n))
    results_df = build_results_df(records)

    csv_out = save_results_csv(results_df, OUT_DIR)
    print(f"\nResults saved → {csv_out}")
    print(results_df.to_string())

    png_out = save_bar_chart(results_df, OUT_DIR)
    print(f"Bar chart saved → {png_out}")

    table_out = save_table_png(results_df, OUT_DIR)
    print(f"Summary table saved → {table_out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
