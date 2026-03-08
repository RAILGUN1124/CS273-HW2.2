"""
Step 3: WordNet-Based Similarity Matrices (GLAO)
=================================================
For each of 4 preprocessing configurations, compute a pairwise similarity
matrix for the 50 attack descriptions using the Greedy Lemma Aligning Overlap
(GLAO) method with WordNet Wu-Palmer similarity.

Configurations (4):
  ┌─────┬──────────────────┬───────────────┐
  │ Cfg │ Stopwords Removed│ Lemmatization │
  ├─────┼──────────────────┼───────────────┤
  │  1  │      No          │      No       │
  │  2  │      No          │      Yes      │
  │  3  │      Yes         │      No       │
  │  4  │      Yes         │      Yes      │
  └─────┴──────────────────┴───────────────┘

Similarity method: Greedy Lemma Aligning Overlap (GLAO)
  sim(d1, d2) = (Σ_{s∈S1} max_{s'∈S2} wup(s,s')
              +  Σ_{s'∈S2} max_{s∈S1} wup(s',s))
              / (|S1| + |S2|)

Output files (saved to step3/):
  - wordnet_cfg{N}_*.csv   : 50×50 similarity matrix for each config
  - all_wordnet_similarity_matrices.png : combined heatmap
"""

import os

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn, stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for resource in ("punkt_tab", "averaged_perceptron_tagger_eng",
                 "wordnet", "stopwords", "omw-1.4"):
    nltk.download(resource, quiet=True)

STOP_WORDS = set(nltk_stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "all_data.xlsx")
OUT_DIR = SCRIPT_DIR

_TB_WN_MAP = {
    "NN": wn.NOUN,
    "VB": wn.VERB,
    "JJ": wn.ADJ,
    "RB": wn.ADV,
}

CONFIGS = [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
]


def load_documents(data_path):
    df_raw = pd.read_excel(data_path, header=0)
    doc_df = df_raw.iloc[:, 2:]
    doc_df = doc_df.dropna(axis=1, how="all")
    doc_df = doc_df.loc[:, doc_df.columns.astype(str).str.strip() != ""]
    return list(doc_df.columns)


def treebank_to_wn(tb_tag):
    return _TB_WN_MAP.get(tb_tag[:2], None)


def extract_synsets(text, remove_stopwords, lemmatize):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if not tokens:
        return []
    synsets = []
    for word, tb_tag in nltk.pos_tag(tokens):
        wn_pos = treebank_to_wn(tb_tag)
        if wn_pos is None:
            continue
        if lemmatize:
            word = LEMMATIZER.lemmatize(word, pos=wn_pos)
            syns = wn.synsets(word, pos=wn_pos)
        else:
            # Exact surface-form lookup: skip WordNet's internal morphy normalization
            syns = [s for s in wn.synsets(word, pos=wn_pos)
                    if word in s.lemma_names()]
        if syns:
            synsets.append(syns[0])
    return synsets


def best_wup(s, candidates):
    best = 0.0
    for c in candidates:
        try:
            score = s.wup_similarity(c)
        except Exception:
            score = None
        if score is not None and score > best:
            best = score
    return best


def glao_similarity(syn1, syn2):
    if not syn1 or not syn2:
        return 0.0
    sum_fwd = sum(best_wup(s, syn2) for s in syn1)
    sum_bwd = sum(best_wup(s, syn1) for s in syn2)
    return (sum_fwd + sum_bwd) / (len(syn1) + len(syn2))


def compute_glao_matrix(documents, remove_sw, lemmatize):
    all_synsets = [extract_synsets(doc, remove_sw, lemmatize) for doc in documents]
    counts = [len(s) for s in all_synsets]
    print(f"  Synset counts — min:{min(counts)}  max:{max(counts)}  mean:{np.mean(counts):.1f}")

    n = len(documents)
    mat = np.zeros((n, n))
    total = n * (n - 1) // 2
    done = 0
    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            score = glao_similarity(all_synsets[i], all_synsets[j])
            mat[i, j] = mat[j, i] = score
            done += 1
            if done % 100 == 0:
                print(f"  progress: {done}/{total}", end="\r", flush=True)
    return mat


def save_matrix_csv(mat, documents, label, out_dir):
    sim_df = pd.DataFrame(mat, index=documents, columns=documents)
    csv_path = os.path.join(out_dir, "csv", f"{label}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    sim_df.to_csv(csv_path)
    return csv_path


def save_individual_heatmaps(results, out_dir):
    individual_dir = os.path.join(out_dir, "png", "individual")
    os.makedirs(individual_dir, exist_ok=True)
    for label, mat in results:
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=ax, label="GLAO sim")
        ax.set_title(label.replace("wordnet_", "").replace("_", "  "), fontsize=10)
        ax.set_xlabel("Document index")
        ax.set_ylabel("Document index")
        plt.tight_layout()
        png_path = os.path.join(individual_dir, f"{label}.png")
        plt.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"  Saved {png_path}")


def save_combined_heatmap(results, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for idx, (label, mat) in enumerate(results):
        ax = axes[idx]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=ax, label="GLAO sim")
        ax.set_title(label.replace("wordnet_", "").replace("_", "  "), fontsize=9)
        ax.set_xlabel("Document index")
        ax.set_ylabel("Document index")
    fig.suptitle(
        "WordNet-Based Similarity Matrices (GLAO / Wu-Palmer)\n"
        "4 preprocessing configurations",
        fontsize=12,
    )
    plt.tight_layout()
    png_path = os.path.join(out_dir, "png", "all_wordnet_similarity_matrices.png")
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"\nCombined heatmap saved → {png_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data …")
    documents = load_documents(DATA_PATH)
    print(f"  {len(documents)} documents loaded.")

    results = []
    for cfg_idx, (remove_sw, lemmatize) in enumerate(CONFIGS, start=1):
        sw_tag = "noSW" if remove_sw else "withSW"
        lem_tag = "lem" if lemmatize else "noLem"
        label = f"wordnet_cfg{cfg_idx:02d}_{sw_tag}_{lem_tag}"

        print(f"\nConfig {cfg_idx}: remove_stopwords={remove_sw}, lemmatize={lemmatize}")
        mat = compute_glao_matrix(documents, remove_sw, lemmatize)
        off_diag = mat[mat < 1]
        print(f"  Done. Off-diag stats — min:{off_diag.min():.4f}  "
              f"max:{off_diag.max():.4f}  mean:{off_diag.mean():.4f}")

        csv_path = save_matrix_csv(mat, documents, label, OUT_DIR)
        print(f"  Saved → {csv_path}")
        results.append((label, mat))

    save_combined_heatmap(results, OUT_DIR)
    save_individual_heatmaps(results, OUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
