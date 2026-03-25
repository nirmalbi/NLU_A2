"""
CSL 7640: Natural Language Understanding - Assignment 2
Script   : visualization.py
Purpose  : Project word embeddings into 2D space using PCA and t-SNE.
           Visualize clusters for both CBOW and Skip-gram models.
           Compare clustering behavior between the two models.

Author   : [Your Name]
Date     : 2026
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from gensim.models import Word2Vec
from sklearn.decomposition import PCA       # for PCA dimensionality reduction
from sklearn.manifold import TSNE           # for t-SNE dimensionality reduction

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

MODELS_DIR          = "models"
CBOW_MODEL_PATH     = os.path.join(MODELS_DIR, "CBOW_dim100_win5_neg5.model")
SKIPGRAM_MODEL_PATH = os.path.join(MODELS_DIR, "Skipgram_dim100_win5_neg5.model")

# Output folder for saving visualizations
VIZ_DIR = "visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# WORD GROUPS FOR VISUALIZATION
# We pick words from different semantic categories
# so we can observe whether Word2Vec clusters them correctly
# ─────────────────────────────────────────────

WORD_GROUPS = {
    "Academic Programs": [
        "phd", "mtech", "bachelors", "masters",
        "postgraduate", "programme", "degree"
    ],
    "Research": [
        "research", "proposal", "presentation",
        "faculty", "synopsis", "thesis"
    ],
    "Courses": [
        "lecture", "course", "lab", "credit",
        "syllabus", "elective", "prerequisite"
    ],
    "Student Life": [
        "student", "examination", "attendance",
        "registration", "grade", "semester"
    ],
    "Technology": [
        "computer", "network", "algorithm",
        "data", "system", "software"
    ]
}

# Colors for each group in the plot
GROUP_COLORS = {
    "Academic Programs" : "#e74c3c",   # red
    "Research"          : "#3498db",   # blue
    "Courses"           : "#2ecc71",   # green
    "Student Life"      : "#f39c12",   # orange
    "Technology"        : "#9b59b6",   # purple
}

# ─────────────────────────────────────────────
# STEP 1: Load models
# ─────────────────────────────────────────────

print("=" * 60)
print("visualization.py — Word Embedding Visualization")
print("CSL 7640 - Assignment 2, Task 4")
print("=" * 60)

print("\n[STEP 1] Loading trained models...")
cbow_model     = Word2Vec.load(CBOW_MODEL_PATH)
skipgram_model = Word2Vec.load(SKIPGRAM_MODEL_PATH)
print("  Models loaded successfully!")


# ─────────────────────────────────────────────
# HELPER: Collect word vectors from model
# ─────────────────────────────────────────────

def collect_vectors(model, word_groups):
    """
    Collects embedding vectors for all words in word_groups
    that exist in the model vocabulary.
    Returns:
        vectors : numpy array of shape (n_words, embedding_dim)
        labels  : list of word strings
        colors  : list of color strings for each word
        groups  : list of group names for each word
    """
    vectors = []
    labels  = []
    colors  = []
    groups  = []

    for group_name, words in word_groups.items():
        for word in words:
            if word in model.wv:
                vectors.append(model.wv[word])  # get embedding vector
                labels.append(word)
                colors.append(GROUP_COLORS[group_name])
                groups.append(group_name)
            else:
                print(f"  Skipping '{word}' — not in vocabulary")

    return np.array(vectors), labels, colors, groups


# ─────────────────────────────────────────────
# HELPER: Plot 2D embeddings
# ─────────────────────────────────────────────

def plot_embeddings(vectors_2d, labels, colors, groups, title, filename):
    """
    Plots 2D word embeddings with colored groups and word labels.
    Saves the plot as a PNG file.
    """
    plt.figure(figsize=(14, 9))

    # Plot each word as a dot
    for i, (label, color) in enumerate(zip(labels, colors)):
        x, y = vectors_2d[i]
        plt.scatter(x, y, color=color, s=100, alpha=0.8, zorder=2)
        plt.annotate(
            label,
            (x, y),
            fontsize=8,
            ha='center',
            va='bottom',
            xytext=(0, 5),
            textcoords='offset points'
        )

    # Create legend for word groups
    legend_patches = [
        mpatches.Patch(color=color, label=group)
        for group, color in GROUP_COLORS.items()
    ]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=9)

    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    filepath = os.path.join(VIZ_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────
# STEP 2: PCA Visualization
# ─────────────────────────────────────────────

print("\n[STEP 2] Generating PCA Visualizations...")

# PCA reduces high-dimensional vectors to 2D by finding
# the directions of maximum variance in the data.
# It's fast and deterministic (same result every run).

pca = PCA(n_components=2, random_state=42)

# --- CBOW PCA ---
print("\n  CBOW + PCA:")
cbow_vectors, cbow_labels, cbow_colors, cbow_groups = collect_vectors(cbow_model, WORD_GROUPS)
cbow_pca_2d = pca.fit_transform(cbow_vectors)
plot_embeddings(
    cbow_pca_2d, cbow_labels, cbow_colors, cbow_groups,
    title    = "PCA Visualization — CBOW Word Embeddings (dim=100)",
    filename = "pca_cbow.png"
)

# --- Skip-gram PCA ---
print("\n  Skip-gram + PCA:")
sg_vectors, sg_labels, sg_colors, sg_groups = collect_vectors(skipgram_model, WORD_GROUPS)
sg_pca_2d = pca.fit_transform(sg_vectors)
plot_embeddings(
    sg_pca_2d, sg_labels, sg_colors, sg_groups,
    title    = "PCA Visualization — Skip-gram Word Embeddings (dim=100)",
    filename = "pca_skipgram.png"
)


# ─────────────────────────────────────────────
# STEP 3: t-SNE Visualization
# ─────────────────────────────────────────────

print("\n[STEP 3] Generating t-SNE Visualizations...")

# t-SNE is a non-linear dimensionality reduction technique.
# It preserves LOCAL structure — nearby points in high-dim space
# stay nearby in 2D. Better for cluster visualization than PCA.
# perplexity controls the balance between local and global structure.
# For small datasets, lower perplexity (5-15) works better.

tsne = TSNE(
    n_components = 2,
    perplexity   = 10,    # lower value for small datasets
    random_state = 42,
    max_iter     = 1000,
    learning_rate= 200
)

# --- CBOW t-SNE ---
print("\n  CBOW + t-SNE:")
cbow_tsne_2d = tsne.fit_transform(cbow_vectors)
plot_embeddings(
    cbow_tsne_2d, cbow_labels, cbow_colors, cbow_groups,
    title    = "t-SNE Visualization — CBOW Word Embeddings (dim=100)",
    filename = "tsne_cbow.png"
)

# --- Skip-gram t-SNE ---
print("\n  Skip-gram + t-SNE:")
sg_tsne_2d = tsne.fit_transform(sg_vectors)
plot_embeddings(
    sg_tsne_2d, sg_labels, sg_colors, sg_groups,
    title    = "t-SNE Visualization — Skip-gram Word Embeddings (dim=100)",
    filename = "tsne_skipgram.png"
)


# ─────────────────────────────────────────────
# STEP 4: Side-by-side comparison plot
# ─────────────────────────────────────────────

print("\n[STEP 4] Generating side-by-side comparison plot...")

fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle(
    "Word Embedding Visualization — CBOW vs Skip-gram\nIIT Jodhpur Corpus",
    fontsize=16, fontweight='bold'
)

# Data to plot: (vectors_2d, labels, colors, title, axis)
plots = [
    (cbow_pca_2d,  cbow_labels, cbow_colors, "PCA — CBOW",      axes[0][0]),
    (sg_pca_2d,    sg_labels,   sg_colors,   "PCA — Skip-gram", axes[0][1]),
    (cbow_tsne_2d, cbow_labels, cbow_colors, "t-SNE — CBOW",    axes[1][0]),
    (sg_tsne_2d,   sg_labels,   sg_colors,   "t-SNE — Skip-gram", axes[1][1]),
]

for vectors_2d, labels, colors, title, ax in plots:
    for i, (label, color) in enumerate(zip(labels, colors)):
        x, y = vectors_2d[i]
        ax.scatter(x, y, color=color, s=80, alpha=0.8)
        ax.annotate(label, (x, y), fontsize=7, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

# Shared legend
legend_patches = [
    mpatches.Patch(color=color, label=group)
    for group, color in GROUP_COLORS.items()
]
fig.legend(handles=legend_patches, loc='lower center', ncol=5,
           fontsize=10, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
comparison_path = os.path.join(VIZ_DIR, "comparison_all.png")
plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {comparison_path}")
plt.show()
plt.close()


# ─────────────────────────────────────────────
# STEP 5: Interpretation
# ─────────────────────────────────────────────

print("\n\n" + "=" * 60)
print("INTERPRETATION — Clustering Behavior")
print("=" * 60)
print("""
  PCA vs t-SNE:
  ─────────────
  - PCA is linear and fast. It captures global structure but
    may not show tight semantic clusters clearly.
  - t-SNE is non-linear and better at revealing local clusters.
    Words with similar embeddings will appear closer together.

  CBOW vs Skip-gram Clustering:
  ──────────────────────────────
  - CBOW tends to produce more uniform clusters because it
    averages context vectors — frequent words dominate.
  - Skip-gram produces more spread-out, distinct clusters
    because it learns from individual word-context pairs.
  - Academic domain words (phd, mtech, research, student)
    should appear in separate but nearby clusters in Skip-gram.
  - Technology words (data, algorithm, network) should form
    their own cluster separate from Student Life words.

  Overall:
  ────────
  - With a small corpus (~37k tokens), clusters may not be
    perfectly separated. A larger corpus would give tighter,
    more semantically meaningful clusters.
  - Skip-gram generally shows better cluster separation for
    rare domain-specific words in our IIT Jodhpur corpus.
""")

print("Task 4 complete ✅")
print("\nFiles saved in 'visualizations/' folder:")
print("  → pca_cbow.png")
print("  → pca_skipgram.png")
print("  → tsne_cbow.png")
print("  → tsne_skipgram.png")
print("  → comparison_all.png")
print("\nQuestion 1 COMPLETE! ✅🎉")