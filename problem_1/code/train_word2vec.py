"""
CSL 7640: Natural Language Understanding - Assignment 2
Script   : train_word2vec.py
Purpose  : Train Word2Vec models (CBOW and Skip-gram with Negative Sampling)
           on the preprocessed IIT Jodhpur corpus.
           Experiments with different embedding dimensions, context window
           sizes, and number of negative samples.

Author   : Nirmal Kumar Godara
"""

import os
import pickle
from gensim.models import Word2Vec   # main Word2Vec implementation
import logging

# Suppress verbose gensim training logs (EPOCH, lifecycle, memory)
logging.getLogger("gensim").setLevel(logging.WARNING)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

SENTENCES_FILE = "sentences.pkl"   # sentence-tokenized corpus from preprocess.py
MODELS_DIR     = "models"          # folder to save trained models
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: Load and prepare corpus
# ─────────────────────────────────────────────

print("=" * 60)
print("train_word2vec.py — Word2Vec Model Training")
print("CSL 7640 - Assignment 2, Task 2")
print("=" * 60)

print("\n[STEP 1] Loading sentences.pkl...")

with open(SENTENCES_FILE, "rb") as f:
    sentences = pickle.load(f)

print(f"  Total sentences : {len(sentences)}")
print(f"  Total tokens    : {sum(len(s) for s in sentences)}")

# ─────────────────────────────────────────────
# STEP 2: Define hyperparameter experiments
# ─────────────────────────────────────────────

"""
We experiment with:
  - Embedding dimension (vector_size): 50, 100, 200
  - Context window size (window)      : 3, 5
  - Negative samples (negative)       : 5, 10

sg = 0 → CBOW (Continuous Bag of Words)
sg = 1 → Skip-gram with Negative Sampling
"""

experiments = [
    # (model_type_name, sg, vector_size, window, negative)
    ("CBOW",      0, 50,  3, 5),
    ("CBOW",      0, 100, 5, 5),
    ("CBOW",      0, 200, 5, 10),
    ("Skipgram",  1, 50,  3, 5),
    ("Skipgram",  1, 100, 5, 5),
    ("Skipgram",  1, 200, 5, 10),
]

# ─────────────────────────────────────────────
# STEP 3: Train all models
# ─────────────────────────────────────────────

print("\n[STEP 2] Training Word2Vec models...\n")

# Store results for printing the summary table
results = []

for model_name, sg, vector_size, window, negative in experiments:

    print(f"\n  Training {model_name} | dim={vector_size} | window={window} | negative={negative}")
    print("  " + "-"*50)

    # Train Word2Vec model
    model = Word2Vec(
        sentences   = sentences,    # our corpus as list of sentences
        vector_size = vector_size,  # size of embedding vectors
        window      = window,       # context window size
        sg          = sg,           # 0=CBOW, 1=Skip-gram
        negative    = negative,     # number of negative samples
        min_count   = 2,            # ignore words appearing less than 2 times
        workers     = 4,            # parallel threads for faster training
        epochs      = 20            # number of training passes over corpus
    )

    # Save model to disk
    model_filename = f"{model_name}_dim{vector_size}_win{window}_neg{negative}.model"
    model_path = os.path.join(MODELS_DIR, model_filename)
    model.save(model_path)

    # Store result for summary table
    results.append({
        "Model"      : model_name,
        "Dim"        : vector_size,
        "Window"     : window,
        "Negative"   : negative,
        "Vocab Size" : len(model.wv),
        "Path"       : model_filename
    })

    print(f"  Saved: {model_path}")
    print(f"  Vocabulary size: {len(model.wv)}")

# ─────────────────────────────────────────────
# STEP 4: Print summary table
# ─────────────────────────────────────────────

print("\n\n" + "=" * 70)
print("TRAINING SUMMARY TABLE")
print("=" * 70)
print(f"{'Model':<12} {'Dim':<6} {'Window':<8} {'Negative':<10} {'Vocab':<8} {'File'}")
print("-" * 70)
for r in results:
    print(f"{r['Model']:<12} {r['Dim']:<6} {r['Window']:<8} {r['Negative']:<10} {r['Vocab Size']:<8} {r['Path']}")
print("=" * 70)

print("\nAll models saved in the 'models/' folder.")
print("Next step: run semantic_analysis.py")