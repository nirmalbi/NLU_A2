"""
CSL 7640: Natural Language Understanding - Assignment 2
Script   : preprocess.py
Purpose  : Takes raw_corpus.txt as input, applies all preprocessing steps,
           saves final_corpus.txt, prints dataset statistics, and generates Word Cloud.

Preprocessing steps :
    (i)   Removal of boilerplate text and formatting artifacts
    (ii)  Tokenization
    (iii) Lowercasing
    (iv)  Removal of excessive punctuation and non-textual content

Author   : Nirmal Kumar Godara
"""

import re                                   # for regex-based text cleaning
import pickle                               # for saving sentences
import nltk                                 # for tokenization and stopwords
from wordcloud import WordCloud             # for generating word cloud
import matplotlib.pyplot as plt            # for displaying the word cloud
from collections import Counter            # for counting word frequencies

# Download required NLTK data 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.corpus import stopwords          # common English stopwords list
from nltk.tokenize import word_tokenize, sent_tokenize

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

INPUT_FILE      = "raw_corpus.txt"    # output from collect_data.py
OUTPUT_FILE     = "final_corpus.txt"  # clean final corpus for word cloud / stats
SENTENCES_FILE  = "sentences.pkl"     # sentence-tokenized corpus for Word2Vec


# ─────────────────────────────────────────────
# STEP 1: Read raw corpus
# ─────────────────────────────────────────────

print("=" * 50)
print("preprocess.py — Corpus Preprocessing")
print("CSL 7640 - Assignment 2, Task 1")
print("=" * 50)

print("\n[STEP 1] Reading raw_corpus.txt...")
with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    raw_text = f.read()

print(f"  Raw text loaded: {len(raw_text.split())} words")


# ─────────────────────────────────────────────
# STEP 2: Remove boilerplate (light clean, keep sentence boundaries)
# ─────────────────────────────────────────────

print("\n[STEP 2] Removing boilerplate (preserving sentence boundaries)...")

text = re.sub(r'---\s*SOURCE:.*?---', '', raw_text)
text = re.sub(r'http\S+|www\S+', '', text)
text = re.sub(r'\S+@\S+', '', text)
text = text.encode('ascii', errors='ignore').decode('ascii')


# ─────────────────────────────────────────────
# STEP 3: Sentence tokenize BEFORE removing punctuation
# ─────────────────────────────────────────────

print("\n[STEP 3] Sentence tokenizing...")
raw_sentences = sent_tokenize(text)
print(f"  Sentences detected: {len(raw_sentences)}")


# ─────────────────────────────────────────────
# STEP 4: Clean each sentence independently
# ─────────────────────────────────────────────

print("\n[STEP 4] Cleaning each sentence (lowercase, remove punct/numbers/stopwords)...")

english_stopwords = set(stopwords.words('english'))

sentences = []
all_clean_tokens = []

for sent in raw_sentences:
    sent = sent.lower()
    sent = re.sub(r'\b\d+\b', '', sent)
    sent = re.sub(r'[^a-zA-Z\s]', ' ', sent)
    tokens = sent.split()
    clean = [
        t for t in tokens
        if t.isalpha() and len(t) > 2 and t not in english_stopwords
    ]
    if len(clean) > 1:
        sentences.append(clean)
        all_clean_tokens.extend(clean)

print(f"  Total sentences (after filtering): {len(sentences)}")
print(f"  Total clean tokens: {len(all_clean_tokens)}")

clean_tokens = all_clean_tokens


# ─────────────────────────────────────────────
# STEP 5: Save sentences.pkl for Word2Vec
# ─────────────────────────────────────────────

print("\n[STEP 5] Saving sentences.pkl...")

with open(SENTENCES_FILE, "wb") as f:
    pickle.dump(sentences, f)

print(f"  Saved to: {SENTENCES_FILE}")


# ─────────────────────────────────────────────
# STEP 6: Save final_corpus.txt (for word cloud / stats)
# ─────────────────────────────────────────────

print("\n[STEP 6] Saving final_corpus.txt...")

final_corpus = " ".join(clean_tokens)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(final_corpus)

print(f"  Saved to: {OUTPUT_FILE}")


# ─────────────────────────────────────────────
# STEP 7: Dataset Statistics (required for report)
# ─────────────────────────────────────────────

print("\n[STEP 7] Computing dataset statistics...")

# Total number of tokens
total_tokens = len(clean_tokens)

# Vocabulary = unique words
vocabulary = set(clean_tokens)
vocab_size  = len(vocabulary)

# Number of documents = number of source files  collected
# (we added --- SOURCE: filename --- markers during collection)
num_documents = raw_text.count("--- SOURCE:")

# Word frequency for top words
word_freq = Counter(clean_tokens)
top_20    = word_freq.most_common(20)

print("\n" + "=" * 50)
print("DATASET STATISTICS")
print("=" * 50)
print(f"  Total documents  : {num_documents}")
print(f"  Total tokens     : {total_tokens}")
print(f"  Vocabulary size  : {vocab_size}")
print("\n  Top 20 most frequent words:")
for word, freq in top_20:
    print(f"    {word:<20} {freq}")
print("=" * 50)


# ─────────────────────────────────────────────
# STEP 8: Word Cloud
# ─────────────────────────────────────────────

print("\n[STEP 8] Generating Word Cloud...")

# Create word cloud from clean tokens
wordcloud = WordCloud(
    width=1200,           # width of the image
    height=600,           # height of the image
    background_color='white',  # white background
    max_words=150,        # show top 150 words
    colormap='viridis',   # color scheme
    collocations=False    # avoid repeating bigrams
).generate(final_corpus)

# Display the word cloud
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # hide axes
plt.title('Word Cloud — IIT Jodhpur Corpus', fontsize=20, fontweight='bold', pad=20)
plt.tight_layout()

# Save word cloud as image
plt.savefig('wordcloud.png', dpi=200, bbox_inches='tight')
print("  Word cloud saved as: wordcloud.png")

# Show the word cloud
plt.show()

print("\nDone! Task 1 complete ")