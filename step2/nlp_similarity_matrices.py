"""
Step 2: NLP Vectorization — BOW & TF-IDF Similarity Matrices
=============================================================
For each of 12 preprocessing configurations × 2 vectorizers (BOW, TF-IDF),
compute a pairwise cosine-similarity matrix over the 51 attack descriptions.
Similarity metric: Cosine similarity
Total outputs: 24 CSV matrices + 1 combined heatmap PNG per run.

Configurations (12):
  Stop-words removed? × Stemming applied? × N-gram range
  ┌─────┬──────────┬─────────┬────────┐
  │ Cfg │ StopW rm │ Stemmed │ N-gram │
  ├─────┼──────────┼─────────┼────────┤
  │  1  │    No    │   No    │  Uni   │
  │  2  │    No    │   No    │  Bi    │
  │  3  │    No    │   No    │  Tri   │
  │  4  │    No    │   Yes   │  Uni   │
  │  5  │    No    │   Yes   │  Bi    │
  │  6  │    No    │   Yes   │  Tri   │
  │  7  │   Yes    │   No    │  Uni   │
  │  8  │   Yes    │   No    │  Bi    │
  │  9  │   Yes    │   No    │  Tri   │
  │ 10  │   Yes    │   Yes   │  Uni   │
  │ 11  │   Yes    │   Yes   │  Bi    │
  │ 12  │   Yes    │   Yes   │  Tri   │
  └─────┴──────────┴─────────┴────────┘
"""

import os
import re
import itertools

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "all_data.xlsx")
OUT_DIR = SCRIPT_DIR

# Supported n-gram ranges and scikit-learn vectorizer classes.
NGRAM_MAP   = {"uni": (1, 1), "bi": (2, 2), "tri": (3, 3)}
VECTORIZERS = {"BOW": CountVectorizer, "TFIDF": TfidfVectorizer}


def load_documents(data_path):
    """Return the attack-description column headers from the Excel sheet."""
    df_raw = pd.read_excel(data_path, header=0)
    # Columns A-B are metadata; C onward are the attack descriptions.
    doc_df = df_raw.iloc[:, 2:]
    doc_df = doc_df.dropna(axis=1, how="all")
    return list(doc_df.columns)


def preprocess(text, remove_stopwords, apply_stemming):
    """Lowercase, strip non-alpha characters, then optionally remove stopwords and stem."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters and spaces
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if apply_stemming:
        tokens = [STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)


def build_config_label(cfg_num, remove_sw, stem, ngram_key):
    """Build a human-readable filename label for a given preprocessing config."""
    sw_tag   = "noSW"   if remove_sw else "withSW"
    stem_tag = "stem"   if stem      else "noStem"
    return f"cfg{cfg_num:02d}_{sw_tag}_{stem_tag}_{ngram_key}gram"


def compute_cosine_matrix(processed_docs, ngram_range, vec_class):
    """Vectorize documents and return their pairwise cosine-similarity matrix.

    Returns (matrix, vocab_size).  If the vocabulary is empty (e.g. all tokens
    were stopwords), returns a zero matrix with ones on the diagonal.
    """
    try:
        vec = vec_class(ngram_range=ngram_range, analyzer="word", min_df=1)
        X = vec.fit_transform(processed_docs)
        return cosine_similarity(X), X.shape[1]  # X.shape[1] == vocab size
    except ValueError as e:
        # Fallback: empty vocabulary — identity-like matrix so downstream code works.
        n = len(processed_docs)
        cos_sim = np.zeros((n, n))
        np.fill_diagonal(cos_sim, 1.0)
        return cos_sim, 0


def save_matrix_csv(cos_sim, documents, label, out_dir):
    """Save a cosine-similarity matrix as a labelled CSV file."""
    sim_df = pd.DataFrame(cos_sim, index=documents, columns=documents)
    csv_path = os.path.join(out_dir, "csv", f"{label}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    sim_df.to_csv(csv_path)
    return csv_path


def compute_all_matrices(documents, out_dir):
    """Iterate over all 12 configs × 2 vectorizers and produce 24 CSV matrices."""
    # Generate all combinations: (remove_sw, stem, ngram) — 2×2×3 = 12 configs.
    configs = list(itertools.product(
        [False, True],        # remove_stopwords
        [False, True],        # apply_stemming
        ["uni", "bi", "tri"]  # n-gram
    ))

    results = []
    for cfg_num, (remove_sw, stem, ngram_key) in enumerate(configs, start=1):
        ngram_range = NGRAM_MAP[ngram_key]
        processed = [preprocess(d, remove_sw, stem) for d in documents]
        tag = build_config_label(cfg_num, remove_sw, stem, ngram_key)

        for vec_name, vec_class in VECTORIZERS.items():
            label = f"{tag}_{vec_name}"
            cos_sim, vocab_size = compute_cosine_matrix(processed, ngram_range, vec_class)
            if vocab_size == 0:
                print(f"  [WARN] {label}: empty vocabulary — filling with zeros.")
            csv_path = save_matrix_csv(cos_sim, documents, label, out_dir)
            results.append((label, cos_sim))
            print(f"  Saved {label}.csv   (vocab size: {vocab_size})")

    return results


def save_individual_heatmaps(results, out_dir):
    """Save one heatmap PNG per matrix into png/individual/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        individual_dir = os.path.join(out_dir, "png", "individual")
        os.makedirs(individual_dir, exist_ok=True)

        for label, mat in results:
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            plt.colorbar(im, ax=ax, label="Cosine similarity")
            ax.set_title(label.replace("_", "  "), fontsize=12)
            ax.set_xlabel("Document index")
            ax.set_ylabel("Document index")
            plt.tight_layout()
            png_path = os.path.join(individual_dir, f"{label}.png")
            plt.savefig(png_path, dpi=120)
            plt.close(fig)
            print(f"  Saved {png_path}")
    except ImportError:
        print("matplotlib not installed — skipping individual heatmaps.")


def save_grouped_heatmaps(results, out_dir):
    """Save 4 grouped PNGs, each a 2×3 grid (rows=vectorizer, cols=n-gram).

    Groups are the 4 (stop-word, stemming) combinations:
      1. withSW_noStem  (cfg01–03)
      2. withSW_stem    (cfg04–06)
      3. noSW_noStem    (cfg07–09)
      4. noSW_stem      (cfg10–12)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        groups = [
            ("withSW_noStem",  "With stop-words, no stemming",  [1, 2, 3]),
            ("withSW_stem",    "With stop-words, stemmed",       [4, 5, 6]),
            ("noSW_noStem",    "Stop-words removed, no stemming",[7, 8, 9]),
            ("noSW_stem",      "Stop-words removed, stemmed",    [10, 11, 12]),
        ]
        ngram_keys  = ["uni", "bi", "tri"]
        vec_names   = ["BOW", "TFIDF"]

        # Build a quick lookup: label -> matrix
        mat_lookup = {label: mat for label, mat in results}

        grouped_dir = os.path.join(out_dir, "png", "grouped")
        os.makedirs(grouped_dir, exist_ok=True)

        for group_tag, group_title, cfg_nums in groups:
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))

            for row_idx, vec_name in enumerate(vec_names):
                for col_idx, (cfg_num, ngram_key) in enumerate(zip(cfg_nums, ngram_keys)):
                    sw_tag   = "noSW"   if "noSW"  in group_tag else "withSW"
                    stem_tag = "stem"   if group_tag.endswith("stem") and not group_tag.endswith("noStem") else "noStem"
                    label = f"cfg{cfg_num:02d}_{sw_tag}_{stem_tag}_{ngram_key}gram_{vec_name}"
                    mat = mat_lookup.get(label)
                    ax = axes[row_idx][col_idx]
                    if mat is not None:
                        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title(f"{vec_name} · {ngram_key}gram", fontsize=12)
                    ax.set_xlabel("Doc index", fontsize=7)
                    ax.set_ylabel("Doc index", fontsize=7)
                    ax.tick_params(labelsize=6)

            plt.tight_layout()
            png_path = os.path.join(grouped_dir, f"group_{group_tag}.png")
            plt.savefig(png_path, dpi=120)
            plt.close(fig)
            print(f"  Grouped heatmap saved → {png_path}")

    except ImportError:
        print("matplotlib not installed — skipping grouped heatmaps.")


def save_combined_heatmap(results, out_dir):
    """Save a single 4×6 grid PNG showing all 24 similarity matrices."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ncols, nrows = 6, 4
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        axes = axes.flatten()

        for idx, (label, mat) in enumerate(results):
            ax = axes[idx]
            ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_title(label.replace("cfg", "C").replace("_", "\n"), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        for idx in range(len(results), len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle(
            "NLP Cosine Similarity Matrices\n"
            "12 preprocessing configs × 2 vectorizers (BOW / TF-IDF)",
            fontsize=12,
        )
        plt.tight_layout()
        png_path = os.path.join(out_dir, "png", "all_nlp_similarity_matrices.png")
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        plt.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"Combined heatmap saved → {png_path}")
    except ImportError:
        print("matplotlib not installed — skipping heatmap.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data …")
    documents = load_documents(DATA_PATH)
    print(f"  {len(documents)} documents loaded.")

    results = compute_all_matrices(documents, OUT_DIR)
    print(f"\nAll {len(results)} matrices saved to {OUT_DIR}/")

    save_combined_heatmap(results, OUT_DIR)
    save_individual_heatmaps(results, OUT_DIR)
    save_grouped_heatmaps(results, OUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
