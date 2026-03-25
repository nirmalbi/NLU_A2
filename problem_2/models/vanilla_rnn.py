"""
vanilla_rnn.py
--------------
Vanilla Recurrent Neural Network (RNN) for character-level Indian name generation.
Implemented from scratch using PyTorch (no high-level RNN wrappers used for the core cell).

Architecture:
    - Input: one-hot encoded character vector
    - Hidden layer: single RNN cell (linear + tanh activation)
    - Output: linear layer over vocabulary → softmax for next-char probability

Author: (Your Name)
Course: CSL 7640 - Natural Language Understanding
Assignment: PA-2, Problem 2, Task 1
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import string
import os

# ─────────────────────────────────────────────
#  1. HYPERPARAMETERS  (experiment with these)
# ─────────────────────────────────────────────
HIDDEN_SIZE   = 256          # Increased hidden units for better capacity
LEARNING_RATE = 0.001        # Reduced LR — prevents overshooting
NUM_EPOCHS    = 80        # More epochs for better convergence
MAX_NAME_LEN  = 20           # Maximum characters a generated name can have
DATA_FILE     = "TrainingNames.txt"   # Path to training data


# ─────────────────────────────────────────────
#  2. DATASET LOADING & VOCABULARY BUILDING
# ─────────────────────────────────────────────

def load_names(filepath):
    """
    Read names from the text file (one name per line).
    Converts everything to lowercase and strips whitespace.
    Returns a list of clean name strings.
    """
    with open(filepath, "r") as f:
        names = [line.strip().lower() for line in f if line.strip()]
    return names


def build_vocab(names):
    """
    Build a character-level vocabulary from all names.
    Special tokens:
        '<' = Start Of Sequence (SOS) — prepended before every name
        '>' = End Of Sequence (EOS)   — appended after every name
    Returns:
        chars     : sorted list of all unique characters + special tokens
        char2idx  : dict mapping char -> integer index
        idx2char  : dict mapping integer index -> char
    """
    # Collect all unique characters appearing in the names
    all_chars = set(c for name in names for c in name)

    # Add special tokens
    special = ['<', '>']  # SOS and EOS
    chars = special + sorted(all_chars)

    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}

    return chars, char2idx, idx2char


# ─────────────────────────────────────────────
#  3. VANILLA RNN MODEL (built from scratch)
# ─────────────────────────────────────────────

class VanillaRNN(nn.Module):
    """
    A single-layer Vanilla RNN cell implemented from scratch.

    At each time step t:
        h_t = tanh( W_ih * x_t  +  W_hh * h_{t-1}  +  b_h )
        y_t = W_ho * h_t + b_o

    Where:
        x_t  = one-hot input vector (vocab_size,)
        h_t  = hidden state vector  (hidden_size,)
        y_t  = output logits        (vocab_size,)
    """

    def __init__(self, vocab_size, hidden_size):
        super(VanillaRNN, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size

        # ── RNN Cell weights ──────────────────────────────────────────────
        # W_ih : maps input  x_t (vocab_size)   → hidden space (hidden_size)
        # W_hh : maps hidden h_{t-1} (hidden_size) → hidden space (hidden_size)
        # Combined into one Linear for efficiency:  input → [x; h] → h
        self.rnn_cell = nn.Linear(vocab_size + hidden_size, hidden_size)

        # ── Output projection ─────────────────────────────────────────────
        # Maps hidden state h_t → logits over vocabulary
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h_prev):
        """
        One step of the RNN.

        Args:
            x      : one-hot input tensor of shape (1, vocab_size)
            h_prev : previous hidden state of shape (1, hidden_size)

        Returns:
            output : raw logits of shape (1, vocab_size)
            h_new  : updated hidden state of shape (1, hidden_size)
        """
        # Concatenate input and previous hidden state along feature dimension
        combined = torch.cat([x, h_prev], dim=1)   # shape: (1, vocab_size + hidden_size)

        # Apply linear transform + tanh non-linearity → new hidden state
        h_new = torch.tanh(self.rnn_cell(combined)) # shape: (1, hidden_size)

        # Project hidden state to vocabulary logits (no softmax here; used in loss)
        output = self.output_layer(h_new)           # shape: (1, vocab_size)

        return output, h_new

    def init_hidden(self):
        """
        Initialize hidden state to zeros at the start of each sequence.
        Shape: (1, hidden_size)
        """
        return torch.zeros(1, self.hidden_size)

    def count_parameters(self):
        """
        Count and return the total number of trainable parameters in the model.
        Reported in Task-1 of the assignment.
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total


# ─────────────────────────────────────────────
#  4. ENCODING HELPERS
# ─────────────────────────────────────────────

def char_to_onehot(char, char2idx, vocab_size):
    """
    Convert a single character to a one-hot encoded tensor.
    Returns tensor of shape (1, vocab_size).
    """
    idx = char2idx[char]
    tensor = torch.zeros(1, vocab_size)
    tensor[0][idx] = 1.0
    return tensor


def name_to_tensors(name, char2idx, vocab_size):
    """
    Convert a full name string into input/target tensor sequences.

    For name "ram":
        Full sequence with SOS/EOS : < r a m >
        Input  sequence (x): < r a m     (drop last)
        Target sequence (y): r a m >     (drop first = SOS)

    Args:
        name      : lowercase name string
        char2idx  : character to index mapping
        vocab_size: total vocabulary size

    Returns:
        input_tensors  : list of one-hot tensors (each shape (1, vocab_size))
        target_indices : list of integer target indices
    """
    # Build full sequence: SOS + characters + EOS
    full_seq = ['<'] + list(name) + ['>']

    # Input = all except last; Target = all except first (SOS)
    input_chars  = full_seq[:-1]
    target_chars = full_seq[1:]

    input_tensors  = [char_to_onehot(c, char2idx, vocab_size) for c in input_chars]
    target_indices = [char2idx[c] for c in target_chars]

    return input_tensors, target_indices


# ─────────────────────────────────────────────
#  5. TRAINING LOOP
# ─────────────────────────────────────────────

def train(model, names, char2idx, vocab_size, num_epochs, lr):
    """
    Train the Vanilla RNN model on the list of names.

    Uses:
        - CrossEntropyLoss at each character step
        - Adam optimizer
        - One name processed per gradient update (online learning)

    Args:
        model      : VanillaRNN instance
        names      : list of training name strings
        char2idx   : character-to-index mapping
        vocab_size : size of the character vocabulary
        num_epochs : number of training epochs
        lr         : learning rate

    Returns:
        loss_history : list of average loss per epoch
    """
    # Adam optimizer — adaptive learning rate, works well for sequence tasks
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # CrossEntropyLoss combines LogSoftmax + NLLLoss
    # It expects raw logits (no softmax needed before this)
    criterion = nn.CrossEntropyLoss()

    loss_history = []

    for epoch in range(1, num_epochs + 1):
        random.shuffle(names)   # Shuffle names each epoch to reduce order bias
        total_loss = 0.0

        for name in names:
            # Skip names with characters not in vocabulary (safety check)
            if any(c not in char2idx for c in name):
                continue

            # Convert name to input/target tensors
            input_tensors, target_indices = name_to_tensors(name, char2idx, vocab_size)

            # Initialize hidden state fresh for each new name (no state carry-over)
            h = model.init_hidden()

            # Zero gradients before each name (online update)
            optimizer.zero_grad()

            loss = torch.tensor(0.0)

            # Process the name one character at a time
            for x_t, target_idx in zip(input_tensors, target_indices):
                output, h = model(x_t, h)   # Forward pass for one timestep

                # Compute loss: compare predicted logits vs true next character
                target_tensor = torch.tensor([target_idx])
                loss = loss + criterion(output, target_tensor)

            # Average loss over characters in this name
            loss = loss / len(input_tensors)

            # Backpropagate
            loss.backward()

            # Gradient clipping — prevents exploding gradients (common in RNNs)
            # Clips gradients so their norm never exceeds 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(names)
        loss_history.append(avg_loss)

        # Print progress every 20 epochs
        if epoch % 20 == 0:
            print(f"Epoch [{epoch:>3}/{num_epochs}]  Avg Loss: {avg_loss:.4f}")

    return loss_history


# ─────────────────────────────────────────────
#  6. NAME GENERATION
# ─────────────────────────────────────────────

def generate_name(model, char2idx, idx2char, vocab_size, max_len=MAX_NAME_LEN, temperature=0.8):
    """
    Generate a single name by sampling from the model character by character.

    Process:
        1. Feed SOS token '<' as the first input
        2. At each step, sample the next character from the output distribution
        3. Stop when EOS '>' is predicted or max_len is reached

    Args:
        model       : trained VanillaRNN
        char2idx    : char to index mapping
        idx2char    : index to char mapping
        vocab_size  : vocabulary size
        max_len     : maximum name length to generate
        temperature : controls randomness
                        < 1.0 → more confident/repetitive
                        > 1.0 → more random/diverse
                        = 1.0 → raw model distribution

    Returns:
        generated name as a string (without SOS/EOS tokens)
    """
    model.eval()  # Switch to eval mode (disables dropout etc. if any)

    with torch.no_grad():   # No gradient tracking needed during generation
        h = model.init_hidden()

        # Start with the SOS token
        input_char = '<'
        generated  = []

        for _ in range(max_len):
            # Encode current character as one-hot
            x = char_to_onehot(input_char, char2idx, vocab_size)

            # Get model output and new hidden state
            output, h = model(x, h)

            # Apply temperature scaling before softmax
            # Dividing logits by temperature adjusts the "peakedness" of distribution
            output = output / temperature

            # Convert logits to probabilities
            probs = torch.softmax(output, dim=1).squeeze()

            # Sample next character index from the probability distribution
            next_idx  = torch.multinomial(probs, num_samples=1).item()
            next_char = idx2char[next_idx]

            # Stop if EOS token is generated
            if next_char == '>':
                break

            # Skip SOS token if accidentally sampled
            if next_char != '<':
                generated.append(next_char)

            input_char = next_char

    # Join characters and capitalize first letter (names convention)
    return ''.join(generated).capitalize()


def generate_names_batch(model, char2idx, idx2char, vocab_size, n=200, temperature=0.8):
    """
    Generate a batch of n names using the trained model.

    Args:
        n           : number of names to generate
        temperature : sampling temperature

    Returns:
        List of generated name strings
    """
    generated = []
    for _ in range(n):
        name = generate_name(model, char2idx, idx2char, vocab_size, temperature=temperature)
        if len(name) > 1:   # Filter out single-character artifacts
            generated.append(name)
    return generated


# ─────────────────────────────────────────────
#  7. MAIN — TRAIN & GENERATE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Load data ──────────────────────────────────────────────────────────
    print("=" * 50)
    print("  Vanilla RNN — Character-Level Name Generation")
    print("=" * 50)

    names = load_names(DATA_FILE)
    print(f"\n[DATA] Loaded {len(names)} names from '{DATA_FILE}'")

    # ── Build vocabulary ───────────────────────────────────────────────────
    chars, char2idx, idx2char = build_vocab(names)
    vocab_size = len(chars)
    print(f"[VOCAB] Vocabulary size: {vocab_size} characters")
    print(f"[VOCAB] Characters: {chars}")

    # ── Initialize model ───────────────────────────────────────────────────
    model = VanillaRNN(vocab_size=vocab_size, hidden_size=HIDDEN_SIZE)

    # Report architecture details (required by Task-1)
    print(f"\n[MODEL] Architecture: Vanilla RNN")
    print(f"[MODEL] Input size   : {vocab_size}  (one-hot vocab size)")
    print(f"[MODEL] Hidden size  : {HIDDEN_SIZE}")
    print(f"[MODEL] Output size  : {vocab_size}  (next-char logits)")
    print(f"[MODEL] Trainable parameters: {model.count_parameters():,}")
    print(f"\n[HYPERPARAMS] Learning Rate : {LEARNING_RATE}")
    print(f"[HYPERPARAMS] Epochs        : {NUM_EPOCHS}")
    print(f"[HYPERPARAMS] Hidden Size   : {HIDDEN_SIZE}")

    # ── Train the model ────────────────────────────────────────────────────
    print(f"\n[TRAINING] Starting training for {NUM_EPOCHS} epochs...\n")
    loss_history = train(model, names, char2idx, vocab_size, NUM_EPOCHS, LEARNING_RATE)
    print(f"\n[TRAINING] Done! Final avg loss: {loss_history[-1]:.4f}")

    # ── Save the trained model ─────────────────────────────────────────────
    torch.save({
        "model_state": model.state_dict(),
        "char2idx"   : char2idx,
        "idx2char"   : idx2char,
        "vocab_size" : vocab_size,
        "hidden_size": HIDDEN_SIZE,
    }, "vanilla_rnn_model.pt")
    print("\n[SAVED] Model saved to 'vanilla_rnn_model.pt'")

    # ── Generate names ─────────────────────────────────────────────────────
    print("\n[GENERATE] Generating 200 names with temperature=0.8 ...\n")
    generated_names = generate_names_batch(model, char2idx, idx2char, vocab_size, n=200)

    # Show 20 sample names on screen
    print("Sample generated names:")
    for i, name in enumerate(generated_names[:20], 1):
        print(f"  {i:>2}. {name}")

    # Save all generated names to output file
    output_path = os.path.join("outputs", "generated_names_rnn.txt")
    os.makedirs("outputs", exist_ok=True)
    with open(output_path, "w") as f:
        for name in generated_names:
            f.write(name + "\n")
    print(f"\n[SAVED] {len(generated_names)} generated names saved to '{output_path}'")