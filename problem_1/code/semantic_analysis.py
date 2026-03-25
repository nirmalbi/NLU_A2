"""
CSL 7640: Natural Language Understanding - Assignment 2
Script   : semantic_analysis.py
Purpose  : Perform semantic analysis on trained Word2Vec models.
           Task 3 requires:
           (1) Top 5 nearest neighbors using cosine similarity for:
               research, student, phd, exam
           (2) At least 3 analogy experiments

           Fix: 'exam' not in vocab → use 'examination'
                'ug','pg' not in vocab → use words that exist

Author   : [Your Name]
Date     : 2026
"""

import os
from gensim.models import Word2Vec

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

MODELS_DIR          = "models"
CBOW_MODEL_PATH     = os.path.join(MODELS_DIR, "CBOW_dim100_win5_neg5.model")
SKIPGRAM_MODEL_PATH = os.path.join(MODELS_DIR, "Skipgram_dim100_win5_neg5.model")

# Assignment requires these 4 words — using closest available vocab words
# 'exam' not in vocab so we use 'examination' which means the same
TARGET_WORDS = ["research", "student", "phd", "examination"]

# Analogy experiments using words confirmed to be in vocabulary
# Format: (word_a, word_b, word_c) → finds word_d
# such that word_a:word_b :: word_c:word_d
ANALOGIES = [
    ("phd", "mtech", "bachelors"),       # PHD:MTech :: Bachelors:?
    ("lecture", "course", "lab"),         # Lecture:Course :: Lab:?
    ("research", "faculty", "student"),   # Research:Faculty :: Student:?
]

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────

print("=" * 60)
print("semantic_analysis.py — Semantic Analysis")
print("CSL 7640 - Assignment 2, Task 3")
print("=" * 60)

print("\n[STEP 1] Loading trained models...")
cbow_model     = Word2Vec.load(CBOW_MODEL_PATH)
skipgram_model = Word2Vec.load(SKIPGRAM_MODEL_PATH)
print(f"  CBOW model loaded     : {CBOW_MODEL_PATH}")
print(f"  Skip-gram model loaded: {SKIPGRAM_MODEL_PATH}")

# ─────────────────────────────────────────────
# HELPER: Check vocab and print available words
# ─────────────────────────────────────────────

def check_vocab(model, words):
    """Check which words from a list exist in the model vocabulary."""
    for w in words:
        status = "✓ IN VOCAB" if w in model.wv else "✗ NOT IN VOCAB"
        print(f"    {w:<20} {status}")

# ─────────────────────────────────────────────
# HELPER: Nearest neighbors
# ─────────────────────────────────────────────

def get_nearest_neighbors(model, word, model_name, topn=5):
    """
    Finds top-n most similar words using cosine similarity.
    Prints results in a formatted table.
    """
    if word not in model.wv:
        # Try plural/singular variations automatically
        alternatives = [word + "s", word + "ion", word[:-1], word[:-3]]
        found = next((w for w in alternatives if w in model.wv), None)
        if found:
            print(f"  [{model_name}] '{word}' not found, using '{found}' instead")
            word = found
        else:
            print(f"  [{model_name}] '{word}' NOT found in vocabulary — skipping")
            return []

    similar = model.wv.most_similar(word, topn=topn)

    print(f"\n  [{model_name}] Top {topn} neighbors for '{word}':")
    print(f"  {'Rank':<6} {'Word':<20} {'Cosine Similarity'}")
    print(f"  {'-'*45}")
    for rank, (sim_word, score) in enumerate(similar, 1):
        print(f"  {rank:<6} {sim_word:<20} {score:.4f}")

    return similar


# ─────────────────────────────────────────────
# STEP 2: Nearest Neighbors
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("TASK 3.1 — TOP 5 NEAREST NEIGHBORS (Cosine Similarity)")
print("=" * 60)

for word in TARGET_WORDS:
    print("\n" + "-"*60)
    print(f"  Query Word: '{word.upper()}'")
    print("-"*60)
    get_nearest_neighbors(cbow_model,     word, "CBOW")
    get_nearest_neighbors(skipgram_model, word, "Skip-gram")


# ─────────────────────────────────────────────
# HELPER: Analogy
# ─────────────────────────────────────────────

def perform_analogy(model, word_a, word_b, word_c, model_name, topn=5):
    """
    Performs word analogy: word_a:word_b :: word_c:?
    Vector arithmetic: result = vec(word_b) - vec(word_a) + vec(word_c)
    """
    print(f"\n  [{model_name}] '{word_a}' : '{word_b}' :: '{word_c}' : ?")

    missing = [w for w in [word_a, word_b, word_c] if w not in model.wv]
    if missing:
        print(f"  Words not in vocabulary: {missing} — skipping")
        return []

    results = model.wv.most_similar(
        positive=[word_b, word_c],
        negative=[word_a],
        topn=topn
    )

    print(f"  {'Rank':<6} {'Word':<20} {'Cosine Similarity'}")
    print(f"  {'-'*45}")
    for rank, (word, score) in enumerate(results, 1):
        print(f"  {rank:<6} {word:<20} {score:.4f}")

    return results


# ─────────────────────────────────────────────
# STEP 3: Analogy Experiments
# ─────────────────────────────────────────────

print("\n\n" + "=" * 60)
print("TASK 3.2 — ANALOGY EXPERIMENTS")
print("=" * 60)

for word_a, word_b, word_c in ANALOGIES:
    print(f"\n{'─'*60}")
    print(f"  Analogy: {word_a.upper()} : {word_b.upper()} :: {word_c.upper()} : ?")
    print(f"{'─'*60}")
    perform_analogy(cbow_model,     word_a, word_b, word_c, "CBOW")
    perform_analogy(skipgram_model, word_a, word_b, word_c, "Skip-gram")


# ─────────────────────────────────────────────
# STEP 4: Discussion
# ─────────────────────────────────────────────

print("\n\n" + "=" * 60)
print("DISCUSSION — Are results semantically meaningful?")
print("=" * 60)
print("""
  1. Nearest Neighbors:
     - 'phd' neighbors (mtech, doctor, preparatory) are semantically meaningful
       as they relate to postgraduate academic programs — good result!
     - 'research' neighbors reflect academic context (faculty, proposal)
     - 'student' neighbors (attendance, candidate) are contextually relevant
     - Skip-gram gives more meaningful rare-word neighbors than CBOW
       because it trains on individual context pairs rather than averaging.

  2. Analogy Experiments:
     - Results depend on co-occurrence patterns in our corpus.
     - With a small domain-specific corpus (~37k tokens), analogies may not
       always produce perfectly expected results.
     - Word2Vec typically needs millions of tokens for reliable analogies.
     - Despite small corpus, domain-specific words show reasonable clustering.

  3. CBOW vs Skip-gram comparison:
     - CBOW: faster training, better for frequent words, high similarity scores
     - Skip-gram: slower but captures rare word semantics better (e.g. 'phd')
     - For our small academic corpus, Skip-gram gives more meaningful results
       for domain-specific terms like 'phd', 'research', 'mtech'
""")

print("Task 3 complete ")
print("Next step: run visualization.py")