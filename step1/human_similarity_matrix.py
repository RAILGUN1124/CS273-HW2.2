"""
Step 1: Human-Based Similarity Matrix
======================================
Reads the card-sorting document vectors from data/all_data.xlsx (columns C–AZ),
then computes the pairwise Spearman rank-correlation similarity matrix.

Output files (saved to step1/):
  - human_similarity_matrix.csv   : 50×50 similarity matrix
  - human_similarity_matrix.png   : heatmap visualisation
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "all_data.xlsx")
OUT_DIR = SCRIPT_DIR


def load_document_vectors(data_path):
    """Load the card-sorting document vectors from the Excel file.

    Returns a DataFrame where each column is one attack description and each
    row is one participant's sorting vector.
    """
    df_raw = pd.read_excel(data_path, header=0)
    # Columns A & B are 'Participant #' and 'Group #'; document vectors start at column C
    doc_df = df_raw.iloc[:, 2:]
    doc_df = doc_df.dropna(axis=1, how="all")
    doc_df = doc_df.loc[:, doc_df.columns.astype(str).str.strip() != ""]
    return doc_df


def compute_spearman_matrix(doc_df):
    """Compute the pairwise Spearman rank-correlation matrix across document columns.

    scipy.stats.spearmanr with axis=0 treats each column as a variable, so the
    result is an n_docs × n_docs correlation matrix.
    """
    vectors = doc_df.values.astype(float)
    corr_matrix, _ = spearmanr(vectors, axis=0)  # correlate columns (documents)
    # spearmanr returns a scalar when there are only 2 documents
    if np.isscalar(corr_matrix):
        corr_matrix = np.array([[1.0, corr_matrix], [corr_matrix, 1.0]])
    return corr_matrix


def save_csv(corr_matrix, doc_labels, out_dir):
    """Save the similarity matrix as a labelled CSV file."""
    sim_df = pd.DataFrame(corr_matrix, index=doc_labels, columns=doc_labels)
    csv_path = os.path.join(out_dir, "csv", "human_similarity_matrix.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    sim_df.to_csv(csv_path)
    print(f"  Saved matrix → {csv_path}")


def save_heatmap(corr_matrix, n_docs, out_dir):
    """Render the similarity matrix as a coolwarm heatmap and save to PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(18, 16))
        im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        plt.colorbar(im, ax=ax, label="Spearman ρ")

        short_labels = [f"D{i+1}" for i in range(n_docs)]
        ax.set_xticks(range(n_docs))
        ax.set_yticks(range(n_docs))
        ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(short_labels, fontsize=7)
        ax.set_title(
            "Human-Based Similarity Matrix\n(Spearman Rank Correlation, card-sorting vectors)",
            fontsize=13,
        )

        plt.tight_layout()
        png_path = os.path.join(out_dir, "png", "human_similarity_matrix.png")
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        plt.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  Saved heatmap → {png_path}")
    except ImportError:
        print("  matplotlib not installed – skipping heatmap.")


def print_sanity_check(corr_matrix, n_docs):
    """Assert diagonal is all 1.0 and print off-diagonal summary statistics."""
    print("\nSanity check – diagonal (should all be 1.0):")
    diag = np.diag(corr_matrix)
    assert np.allclose(diag, 1.0), "Diagonal is not all 1.0!"
    print(f"  min={diag.min():.6f}  max={diag.max():.6f}  ✓")

    print("\nOff-diagonal statistics:")
    off = corr_matrix[~np.eye(n_docs, dtype=bool)]
    print(f"  min  = {off.min():.4f}")
    print(f"  max  = {off.max():.4f}")
    print(f"  mean = {off.mean():.4f}")
    print(f"  std  = {off.std():.4f}")


def main():
    print("Loading data …")
    doc_df = load_document_vectors(DATA_PATH)
    n_docs = doc_df.shape[1]
    doc_labels = list(doc_df.columns)
    print(f"  {n_docs} documents found, {doc_df.shape[0]} participants.")

    print("Computing pairwise Spearman rank correlations …")
    corr_matrix = compute_spearman_matrix(doc_df)

    save_csv(corr_matrix, doc_labels, OUT_DIR)
    save_heatmap(corr_matrix, n_docs, OUT_DIR)
    print_sanity_check(corr_matrix, n_docs)

    print("\nDone.")


if __name__ == "__main__":
    main()
