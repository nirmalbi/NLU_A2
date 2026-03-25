"""
blstm_rnn.py
------------
Bidirectional LSTM (BLSTM) for character-level Indian name generation.

Architecture:
    - Input     : one-hot encoded character vector (vocab_size,)
    - BLSTM     : bidirectional LSTM processes full sequence (training only)
                  forward hidden states  → size hidden_size
                  backward hidden states → size hidden_size
                  concatenated           → size 2 * hidden_size
    - Two output heads trained jointly:
        output_layer     : (2 * hidden_size) → vocab_size   [training loss]
        gen_output_layer : (hidden_size)     → vocab_size   [generation + aux loss]
    - After training, forward weights are copied into a standalone LSTMCell
      for autoregressive generation.

Author: (Your Name)
Course: CSL 7640 - Natural Language Understanding
Assignment: PA-2, Problem 2, Task 1
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# ─────────────────────────────────────────────
#  1. HYPERPARAMETERS
# ─────────────────────────────────────────────
HIDDEN_SIZE   = 128         # Per direction (reduced from 256 to prevent overfitting)
LEARNING_RATE = 0.001
NUM_EPOCHS    = 80
MAX_NAME_LEN  = 20
DROPOUT_RATE  = 0.3         # Dropout on BLSTM output to regularise
DATA_FILE     = "TrainingNames.txt"


# ─────────────────────────────────────────────
#  2. DATA LOADING & VOCABULARY
# ─────────────────────────────────────────────

def load_names(filepath):
    with open(filepath, "r") as f:
        names = [line.strip().lower() for line in f if line.strip()]
    return names


def build_vocab(names):
    all_chars = set(c for name in names for c in name)
    chars     = ['<', '>'] + sorted(all_chars)   # SOS, EOS + alphabet
    char2idx  = {c: i for i, c in enumerate(chars)}
    idx2char  = {i: c for c, i in char2idx.items()}
    return chars, char2idx, idx2char


# ─────────────────────────────────────────────
#  3. BLSTM MODEL
# ─────────────────────────────────────────────

class BLSTMNameGenerator(nn.Module):
    """
    Bidirectional LSTM for character-level name modelling.

    LSTM gates (per direction, per time step):
        f_t = sigmoid( W_f . [h_{t-1}, x_t] + b_f )      <- Forget gate
        i_t = sigmoid( W_i . [h_{t-1}, x_t] + b_i )      <- Input  gate
        g_t = tanh(    W_g . [h_{t-1}, x_t] + b_g )      <- Candidate cell values
        c_t = f_t * c_{t-1} + i_t * g_t                  <- Cell state update
        o_t = sigmoid( W_o . [h_{t-1}, x_t] + b_o )      <- Output gate
        h_t = o_t * tanh(c_t)                             <- Hidden state

    Bidirectional:
        h_fwd : forward pass   (left  -> right)
        h_bwd : backward pass  (right -> left)
        concat: [h_fwd ; h_bwd] size = 2 * hidden_size  (used for training loss)
        h_fwd alone: size = hidden_size                  (used for generation)

    Two output heads are trained jointly so that BOTH are optimised:
        output_layer     takes the full concatenated context  -> training loss
        gen_output_layer takes only the forward hidden states -> aux loss + generation
    """

    def __init__(self, vocab_size, hidden_size, dropout_rate=0.3):
        super(BLSTMNameGenerator, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size

        # Bidirectional LSTM
        # batch_first=True  -> input/output shaped (batch, seq_len, features)
        # bidirectional=True -> creates forward + backward cells automatically
        self.blstm = nn.LSTM(
            input_size    = vocab_size,
            hidden_size   = hidden_size,
            num_layers    = 1,
            bidirectional = True,
            batch_first   = True
        )

        # Dropout applied to BLSTM output to prevent overfitting
        self.dropout = nn.Dropout(p=dropout_rate)

        # Output head 1: full bidirectional context (training)
        # Input : 2 * hidden_size (forward + backward concatenated)
        self.output_layer = nn.Linear(2 * hidden_size, vocab_size)

        # Output head 2: forward-only context (generation + aux loss)
        # CRITICAL: must be trained jointly during training so that
        # generation (which can only use the forward direction) is meaningful.
        # We extract forward hidden states (first hidden_size features of BLSTM output)
        # and train this head on those — so it's ready at generation time.
        self.gen_output_layer = nn.Linear(hidden_size, vocab_size)

        # Standalone forward LSTMCell — used ONLY during generation (step-by-step).
        # After training, we copy the forward-direction weights from blstm into here.
        self.forward_cell = nn.LSTMCell(
            input_size  = vocab_size,
            hidden_size = hidden_size
        )

    def forward(self, x_seq):
        """
        Full-sequence forward pass — TRAINING ONLY.

        Returns logits from BOTH output heads so both get trained.

        Args:
            x_seq : shape (1, seq_len, vocab_size)

        Returns:
            full_logits : (seq_len, vocab_size) from output_layer
            fwd_logits  : (seq_len, vocab_size) from gen_output_layer
        """
        # Run bidirectional LSTM over the full input sequence
        # lstm_out: (1, seq_len, 2*hidden_size)
        # First hidden_size features  = forward  direction outputs
        # Last  hidden_size features  = backward direction outputs
        lstm_out, _ = self.blstm(x_seq)

        # Apply dropout for regularisation
        lstm_out = self.dropout(lstm_out)

        # Remove batch dimension -> (seq_len, 2 * hidden_size)
        lstm_out = lstm_out.squeeze(0)

        # Head 1: full bidirectional logits
        full_logits = self.output_layer(lstm_out)

        # Head 2: forward-only logits (extract first hidden_size columns)
        fwd_out    = lstm_out[:, :self.hidden_size]
        fwd_logits = self.gen_output_layer(fwd_out)

        return full_logits, fwd_logits

    def sync_forward_cell_weights(self):
        """
        Copy trained forward-direction weights from self.blstm into self.forward_cell.

        PyTorch LSTM weight names (forward direction):
            weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
        """
        with torch.no_grad():
            self.forward_cell.weight_ih.copy_(self.blstm.weight_ih_l0)
            self.forward_cell.weight_hh.copy_(self.blstm.weight_hh_l0)
            self.forward_cell.bias_ih.copy_(self.blstm.bias_ih_l0)
            self.forward_cell.bias_hh.copy_(self.blstm.bias_hh_l0)

    def generate_step(self, x, h, c):
        """
        Single autoregressive step — GENERATION ONLY.
        Uses the forward LSTMCell + gen_output_layer (both fully trained).

        Args:
            x : one-hot input (1, vocab_size)
            h : forward hidden state (1, hidden_size)
            c : forward cell  state  (1, hidden_size)

        Returns: logits (1, vocab_size), h_new, c_new
        """
        h_new, c_new = self.forward_cell(x, (h, c))
        logits = self.gen_output_layer(h_new)
        return logits, h_new, c_new

    def init_hidden_cell(self):
        """Return zeroed (h, c) for the forward LSTMCell."""
        return torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
#  4. ENCODING HELPERS
# ─────────────────────────────────────────────

def char_to_onehot(char, char2idx, vocab_size):
    """Return one-hot tensor of shape (1, vocab_size)."""
    t = torch.zeros(1, vocab_size)
    t[0][char2idx[char]] = 1.0
    return t


def name_to_sequence_tensor(name, char2idx, vocab_size):
    """
    Convert a name to input tensor + target indices.

    For "ram":  full = [<, r, a, m, >]
        x_seq   = one-hot of [<, r, a, m]  -> shape (1, 4, vocab_size)
        targets = indices of [r, a, m, >]  -> list of 4 ints
    """
    full_seq = ['<'] + list(name) + ['>']
    inputs   = full_seq[:-1]
    targets  = full_seq[1:]

    onehots = [char_to_onehot(c, char2idx, vocab_size) for c in inputs]
    x_seq   = torch.cat(onehots, dim=0).unsqueeze(0)   # (1, seq_len, vocab_size)
    t_idx   = [char2idx[c] for c in targets]

    return x_seq, t_idx


# ─────────────────────────────────────────────
#  5. TRAINING LOOP
# ─────────────────────────────────────────────

def train(model, names, char2idx, vocab_size, num_epochs, lr):
    """
    Train BLSTM with combined loss from both output heads.

    Loss = CrossEntropy(full_logits, targets)   <- trains output_layer + BLSTM
         + CrossEntropy(fwd_logits,  targets)   <- trains gen_output_layer (KEY FIX)

    Training gen_output_layer jointly ensures the generation path is
    optimised and doesn't remain at random initialisation.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loss_history = []

    for epoch in range(1, num_epochs + 1):
        random.shuffle(names)
        total_loss = 0.0

        for name in names:
            if any(c not in char2idx for c in name):
                continue

            x_seq, t_idx = name_to_sequence_tensor(name, char2idx, vocab_size)
            targets       = torch.tensor(t_idx)

            optimizer.zero_grad()

            # Both heads forward
            full_logits, fwd_logits = model(x_seq)

            # Combined loss — both heads backpropagate into shared BLSTM weights
            loss = criterion(full_logits, targets) + criterion(fwd_logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(names)
        loss_history.append(avg_loss)

        if epoch % 20 == 0:
            print(f"Epoch [{epoch:>3}/{num_epochs}]  Avg Loss: {avg_loss:.4f}")

    return loss_history


# ─────────────────────────────────────────────
#  6. GENERATION
# ─────────────────────────────────────────────

def generate_name(model, char2idx, idx2char, vocab_size, max_len=MAX_NAME_LEN, temperature=0.8):
    """
    Autoregressively generate one name using the forward LSTMCell.
    Starts from SOS '<', samples until EOS '>' or max_len reached.
    """
    model.eval()
    with torch.no_grad():
        h, c = model.init_hidden_cell()
        input_char = '<'
        generated  = []

        for _ in range(max_len):
            x = char_to_onehot(input_char, char2idx, vocab_size)
            logits, h, c = model.generate_step(x, h, c)

            logits = logits / temperature
            probs  = torch.softmax(logits, dim=1).squeeze()
            next_idx  = torch.multinomial(probs, num_samples=1).item()
            next_char = idx2char[next_idx]

            if next_char == '>':
                break
            if next_char != '<':
                generated.append(next_char)
            input_char = next_char

    return ''.join(generated).capitalize()


def generate_names_batch(model, char2idx, idx2char, vocab_size, n=200, temperature=0.8):
    generated = []
    for _ in range(n):
        name = generate_name(model, char2idx, idx2char, vocab_size, temperature=temperature)
        if len(name) > 1:
            generated.append(name)
    return generated


# ─────────────────────────────────────────────
#  7. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("  BLSTM -- Bidirectional LSTM Name Generation")
    print("=" * 55)

    names = load_names(DATA_FILE)
    print(f"\n[DATA] Loaded {len(names)} names from '{DATA_FILE}'")

    chars, char2idx, idx2char = build_vocab(names)
    vocab_size = len(chars)
    print(f"[VOCAB] Size: {vocab_size}   Characters: {chars}")

    model = BLSTMNameGenerator(vocab_size, HIDDEN_SIZE, DROPOUT_RATE)

    print(f"\n[MODEL] Architecture     : Bidirectional LSTM (BLSTM)")
    print(f"[MODEL] Hidden size      : {HIDDEN_SIZE} per direction  ({2*HIDDEN_SIZE} combined)")
    print(f"[MODEL] Dropout          : {DROPOUT_RATE}")
    print(f"[MODEL] Trainable params : {model.count_parameters():,}")
    print(f"[HYPERPARAMS] LR={LEARNING_RATE}  Epochs={NUM_EPOCHS}  Hidden={HIDDEN_SIZE}")

    print(f"\n[TRAINING] Starting...\n")
    loss_history = train(model, names, char2idx, vocab_size, NUM_EPOCHS, LEARNING_RATE)
    print(f"\n[TRAINING] Done. Final avg loss: {loss_history[-1]:.4f}")

    # Copy forward BLSTM weights into standalone forward cell for generation
    model.sync_forward_cell_weights()
    print("[SYNC] Forward cell weights synced from trained BLSTM.")

    torch.save({
        "model_state" : model.state_dict(),
        "char2idx"    : char2idx,
        "idx2char"    : idx2char,
        "vocab_size"  : vocab_size,
        "hidden_size" : HIDDEN_SIZE,
    }, "blstm_model.pt")
    print("[SAVED] Model saved to 'blstm_model.pt'")

    print("\n[GENERATE] Generating 200 names...\n")
    generated_names = generate_names_batch(model, char2idx, idx2char, vocab_size, n=200)

    print("Sample generated names:")
    for i, name in enumerate(generated_names[:20], 1):
        print(f"  {i:>2}. {name}")

    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "generated_names_blstm.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(generated_names))
    print(f"\n[SAVED] {len(generated_names)} names saved to '{output_path}'")