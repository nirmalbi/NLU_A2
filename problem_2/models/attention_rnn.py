"""
attention_rnn.py
----------------
Attention-based RNN for character-level Indian name generation.
Implemented from scratch using PyTorch.

Architecture:
    - Input     : one-hot encoded character vector (vocab_size,)
    - Encoder   : Vanilla RNN cell processes the full input sequence
                  → produces a sequence of hidden states (context vectors)
    - Attention : At each decoder step, computes attention scores over all
                  encoder hidden states using a learned alignment model
                  → weighted sum = context vector c_t
    - Decoder   : Single RNN step conditioned on [prev_char ; c_t]
                  → predicts next character

    During TRAINING  : teacher forcing — encoder sees full input,
                       decoder uses ground-truth prev chars step by step.
    During GENERATION: autoregressive — encoder re-runs on the growing
                       prefix each step (simple & clean for short names).

Attention mechanism (Bahdanau-style additive):
    e_t,i = v^T  tanh( W_enc * h_enc_i  +  W_dec * h_dec_{t-1} )
    α_t   = softmax( e_t )                 (weights over encoder states)
    c_t   = Σ_i  α_t,i * h_enc_i          (context vector)

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
HIDDEN_SIZE   = 256          # Hidden units for encoder & decoder RNNs
ATTN_SIZE     = 128          # Internal size of the attention alignment layer
LEARNING_RATE = 0.001
NUM_EPOCHS    = 80
MAX_NAME_LEN  = 20
DATA_FILE     = "TrainingNames.txt"


# ─────────────────────────────────────────────
#  2. DATASET LOADING & VOCABULARY BUILDING
# ─────────────────────────────────────────────

def load_names(filepath):
    """Read names from file — one per line, lowercased."""
    with open(filepath, "r") as f:
        names = [line.strip().lower() for line in f if line.strip()]
    return names


def build_vocab(names):
    """
    Build character vocabulary.
    Special tokens:
        '<' = SOS (Start of Sequence)
        '>' = EOS (End of Sequence)
    """
    all_chars = set(c for name in names for c in name)
    chars     = ['<', '>'] + sorted(all_chars)
    char2idx  = {c: i for i, c in enumerate(chars)}
    idx2char  = {i: c for c, i in char2idx.items()}
    return chars, char2idx, idx2char


# ─────────────────────────────────────────────
#  3. ATTENTION RNN MODEL
# ─────────────────────────────────────────────

class AttentionRNN(nn.Module):
    """
    Attention-based RNN for character-level sequence generation.

    Components
    ----------
    Encoder RNN cell:
        h_enc_t = tanh( W_enc( [x_t ; h_enc_{t-1}] ) )

    Attention alignment (Bahdanau additive):
        e_t,i   = v^T tanh( W_enc_a * h_enc_i  +  W_dec_a * h_dec )
        α_t     = softmax( e_t )
        c_t     = Σ_i α_t,i * h_enc_i

    Decoder RNN cell (conditioned on context):
        h_dec_t = tanh( W_dec( [x_t ; c_t ; h_dec_{t-1}] ) )

    Output projection:
        y_t     = W_out * h_dec_t       (raw logits → softmax externally)
    """

    def __init__(self, vocab_size, hidden_size, attn_size):
        super(AttentionRNN, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size
        self.attn_size   = attn_size

        # ── Encoder RNN cell ──────────────────────────────────────────────
        # Concatenates [x_t (vocab_size) ; h_prev (hidden_size)] → h_new
        self.encoder_cell = nn.Linear(vocab_size + hidden_size, hidden_size)

        # ── Attention alignment network ───────────────────────────────────
        # Projects encoder hidden state → attn_size
        self.attn_W_enc = nn.Linear(hidden_size, attn_size, bias=False)
        # Projects decoder hidden state → attn_size
        self.attn_W_dec = nn.Linear(hidden_size, attn_size, bias=False)
        # Scores each (encoder, decoder) pair → scalar
        self.attn_v     = nn.Linear(attn_size, 1, bias=False)

        # ── Decoder RNN cell ──────────────────────────────────────────────
        # Input: [x_t (vocab_size) ; context c_t (hidden_size) ; h_dec_prev (hidden_size)]
        self.decoder_cell = nn.Linear(vocab_size + hidden_size + hidden_size, hidden_size)

        # ── Output projection ─────────────────────────────────────────────
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    # ── Encoder ───────────────────────────────────────────────────────────

    def encode(self, input_tensors):
        """
        Run the encoder over a full sequence of one-hot character tensors.

        Args:
            input_tensors : list of tensors, each shape (1, vocab_size)

        Returns:
            encoder_states : tensor of shape (seq_len, hidden_size)
                             — all encoder hidden states stacked
            h_enc          : final encoder hidden state (1, hidden_size)
        """
        h_enc = self.init_hidden()
        states = []

        for x_t in input_tensors:
            combined = torch.cat([x_t, h_enc], dim=1)   # (1, vocab_size + hidden_size)
            h_enc    = torch.tanh(self.encoder_cell(combined))   # (1, hidden_size)
            states.append(h_enc)

        # Stack all hidden states: (seq_len, hidden_size)
        encoder_states = torch.cat(states, dim=0)
        return encoder_states, h_enc

    # ── Attention ──────────────────────────────────────────────────────────

    def attend(self, encoder_states, h_dec):
        """
        Compute attention context vector for one decoder step.

        Bahdanau additive attention:
            e_i   = v^T  tanh( W_enc * enc_i  +  W_dec * h_dec )
            α     = softmax(e)
            c     = Σ_i α_i * enc_i

        Args:
            encoder_states : (seq_len, hidden_size)
            h_dec          : (1, hidden_size)  — current decoder hidden state

        Returns:
            context   : (1, hidden_size)   — weighted sum of encoder states
            attn_w    : (seq_len,)          — attention weights (for inspection)
        """
        seq_len = encoder_states.shape[0]

        # Project encoder states: (seq_len, attn_size)
        enc_proj = self.attn_W_enc(encoder_states)

        # Project decoder state and expand: (seq_len, attn_size)
        dec_proj = self.attn_W_dec(h_dec).expand(seq_len, -1)

        # Energy scores: (seq_len, 1) → (seq_len,)
        energy   = self.attn_v(torch.tanh(enc_proj + dec_proj)).squeeze(1)

        # Normalise to get attention weights
        attn_w   = torch.softmax(energy, dim=0)      # (seq_len,)

        # Weighted sum of encoder states → context vector
        # encoder_states : (seq_len, hidden_size)
        # attn_w         : (seq_len,) → unsqueeze to (seq_len, 1) for broadcast
        context = (attn_w.unsqueeze(1) * encoder_states).sum(dim=0, keepdim=True)

        return context, attn_w

    # ── Decoder step ───────────────────────────────────────────────────────

    def decode_step(self, x_t, h_dec, encoder_states):
        """
        One step of the attention decoder.

        Args:
            x_t            : one-hot input (1, vocab_size)
            h_dec          : previous decoder hidden state (1, hidden_size)
            encoder_states : all encoder hidden states (seq_len, hidden_size)

        Returns:
            output  : raw logits (1, vocab_size)
            h_dec   : new decoder hidden state (1, hidden_size)
            attn_w  : attention weights (seq_len,)
        """
        # 1. Compute context vector via attention
        context, attn_w = self.attend(encoder_states, h_dec)

        # 2. Concatenate [input ; context ; prev_hidden] → decoder input
        combined = torch.cat([x_t, context, h_dec], dim=1)

        # 3. Decoder RNN step
        h_dec = torch.tanh(self.decoder_cell(combined))

        # 4. Project to vocabulary logits
        output = self.output_layer(h_dec)

        return output, h_dec, attn_w

    # ── Full forward pass (TRAINING) ───────────────────────────────────────

    def forward(self, input_tensors, target_input_tensors):
        """
        Full sequence forward pass — used during TRAINING with teacher forcing.

        Teacher forcing: the decoder receives the ground-truth previous
        character (not its own prediction) at each step. This stabilises
        training significantly.

        Args:
            input_tensors        : list of one-hot tensors for the ENCODER
                                   (the full name sequence, SOS → last char)
            target_input_tensors : list of one-hot tensors for the DECODER
                                   (same sequence — decoder sees each char,
                                    predicts the next one)

        Returns:
            outputs : list of raw logit tensors, each shape (1, vocab_size)
        """
        # Encode the full input sequence
        encoder_states, h_enc = self.encode(input_tensors)

        # Initialise decoder hidden state from final encoder state
        h_dec = h_enc

        outputs = []

        # Decode step by step with teacher forcing
        for x_t in target_input_tensors:
            output, h_dec, _ = self.decode_step(x_t, h_dec, encoder_states)
            outputs.append(output)

        return outputs

    # ── Helpers ────────────────────────────────────────────────────────────

    def init_hidden(self):
        """Return zeroed initial hidden state (1, hidden_size)."""
        return torch.zeros(1, self.hidden_size)

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
#  4. ENCODING HELPERS
# ─────────────────────────────────────────────

def char_to_onehot(char, char2idx, vocab_size):
    """Convert a single character to a one-hot tensor of shape (1, vocab_size)."""
    t = torch.zeros(1, vocab_size)
    t[0][char2idx[char]] = 1.0
    return t


def name_to_tensors(name, char2idx, vocab_size):
    """
    Convert a name to encoder input, decoder input, and target tensors.

    For name "ram":
        full sequence      : < r a m >
        encoder input (x)  : < r a m      (drop last)
        decoder input (d)  : < r a m      (same as encoder — teacher forcing)
        target indices (y) : r a m >      (drop SOS)

    Returns:
        enc_tensors  : list of one-hot tensors for the encoder
        dec_tensors  : list of one-hot tensors for the decoder (teacher forcing)
        target_idx   : list of integer target character indices
    """
    full_seq = ['<'] + list(name) + ['>']

    enc_chars    = full_seq[:-1]   # encoder input
    dec_chars    = full_seq[:-1]   # decoder input (teacher forcing = same)
    target_chars = full_seq[1:]    # prediction targets

    enc_tensors = [char_to_onehot(c, char2idx, vocab_size) for c in enc_chars]
    dec_tensors = [char_to_onehot(c, char2idx, vocab_size) for c in dec_chars]
    target_idx  = [char2idx[c] for c in target_chars]

    return enc_tensors, dec_tensors, target_idx


# ─────────────────────────────────────────────
#  5. TRAINING LOOP
# ─────────────────────────────────────────────

def train(model, names, char2idx, vocab_size, num_epochs, lr):
    """
    Train the Attention RNN with teacher forcing.

    Loss: CrossEntropyLoss over all character prediction steps.
    Optimizer: Adam with gradient clipping (norm ≤ 1.0).

    Args:
        model      : AttentionRNN instance
        names      : list of training name strings
        char2idx   : char-to-index mapping
        vocab_size : vocabulary size
        num_epochs : total training epochs
        lr         : learning rate

    Returns:
        loss_history : list of average loss per epoch
    """
    optimizer    = optim.Adam(model.parameters(), lr=lr)
    criterion    = nn.CrossEntropyLoss()
    loss_history = []

    for epoch in range(1, num_epochs + 1):
        random.shuffle(names)
        total_loss = 0.0

        for name in names:
            if any(c not in char2idx for c in name):
                continue

            enc_tensors, dec_tensors, target_idx = name_to_tensors(
                name, char2idx, vocab_size
            )

            optimizer.zero_grad()

            # Forward pass: encoder + attention decoder (teacher forcing)
            outputs = model(enc_tensors, dec_tensors)

            # Accumulate cross-entropy loss over all timesteps
            loss = torch.tensor(0.0)
            for output, tgt in zip(outputs, target_idx):
                loss = loss + criterion(output, torch.tensor([tgt]))

            # Average over sequence length
            loss = loss / len(outputs)

            loss.backward()

            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(names)
        loss_history.append(avg_loss)

        if epoch % 20 == 0:
            print(f"Epoch [{epoch:>3}/{num_epochs}]  Avg Loss: {avg_loss:.4f}")

    return loss_history


# ─────────────────────────────────────────────
#  6. NAME GENERATION
# ─────────────────────────────────────────────

def generate_name(model, char2idx, idx2char, vocab_size,
                  max_len=MAX_NAME_LEN, temperature=0.8):
    """
    Autoregressively generate a single name using the trained model.

    Generation procedure (attention RNN):
        At each step t:
          1. Re-encode the growing prefix (chars generated so far + SOS)
          2. Compute attention over all encoder states
          3. Decoder predicts the next character
          4. Sample from the temperature-scaled distribution
          5. Append to the name; stop on EOS '>' or max_len

    Re-encoding the prefix at every step is simple and works well for
    short sequences like names. No separate inference-time state management
    is needed.

    Args:
        model       : trained AttentionRNN
        char2idx    : char to index mapping
        idx2char    : index to char mapping
        vocab_size  : vocabulary size
        max_len     : maximum characters to generate
        temperature : sampling temperature
                        < 1.0 → more confident/peaked
                        > 1.0 → more random/diverse

    Returns:
        generated name string (capitalised, without SOS/EOS)
    """
    model.eval()

    with torch.no_grad():
        generated = []
        input_char = '<'

        for _ in range(max_len):
            # Build encoder input: SOS + all generated chars so far
            prefix      = ['<'] + generated
            enc_tensors = [char_to_onehot(c, char2idx, vocab_size) for c in prefix]

            # Encode the current prefix
            encoder_states, h_enc = model.encode(enc_tensors)

            # Decoder starts from final encoder hidden state
            h_dec = h_enc

            # Run decoder attention step for the LAST character in the prefix
            # (we want to predict what comes next after the prefix)
            x_t           = char_to_onehot(input_char, char2idx, vocab_size)
            output, _, _  = model.decode_step(x_t, h_dec, encoder_states)

            # Temperature-scaled sampling
            output    = output / temperature
            probs     = torch.softmax(output, dim=1).squeeze()
            next_idx  = torch.multinomial(probs, num_samples=1).item()
            next_char = idx2char[next_idx]

            # Stop on EOS
            if next_char == '>':
                break

            # Skip accidental SOS samples
            if next_char != '<':
                generated.append(next_char)

            input_char = next_char

    return ''.join(generated).capitalize()


def generate_names_batch(model, char2idx, idx2char, vocab_size,
                         n=200, temperature=0.8):
    """Generate a batch of n names; filter out single-character results."""
    generated = []
    for _ in range(n):
        name = generate_name(model, char2idx, idx2char, vocab_size,
                             temperature=temperature)
        if len(name) > 1:
            generated.append(name)
    return generated


# ─────────────────────────────────────────────
#  7. MAIN — TRAIN & GENERATE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("  Attention RNN -- Character-Level Name Generation")
    print("=" * 55)

    # ── Load data ──────────────────────────────────────────────────────────
    names = load_names(DATA_FILE)
    print(f"\n[DATA] Loaded {len(names)} names from '{DATA_FILE}'")

    # ── Build vocabulary ───────────────────────────────────────────────────
    chars, char2idx, idx2char = build_vocab(names)
    vocab_size = len(chars)
    print(f"[VOCAB] Size: {vocab_size}   Characters: {chars}")

    # ── Initialize model ───────────────────────────────────────────────────
    model = AttentionRNN(vocab_size=vocab_size,
                         hidden_size=HIDDEN_SIZE,
                         attn_size=ATTN_SIZE)

    print(f"\n[MODEL] Architecture       : Attention RNN (Bahdanau)")
    print(f"[MODEL] Input size         : {vocab_size}  (one-hot vocab size)")
    print(f"[MODEL] Hidden size        : {HIDDEN_SIZE}  (encoder & decoder)")
    print(f"[MODEL] Attention size     : {ATTN_SIZE}")
    print(f"[MODEL] Output size        : {vocab_size}  (next-char logits)")
    print(f"[MODEL] Trainable params   : {model.count_parameters():,}")
    print(f"\n[HYPERPARAMS] Learning Rate : {LEARNING_RATE}")
    print(f"[HYPERPARAMS] Epochs        : {NUM_EPOCHS}")
    print(f"[HYPERPARAMS] Hidden Size   : {HIDDEN_SIZE}")
    print(f"[HYPERPARAMS] Attn Size     : {ATTN_SIZE}")

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\n[TRAINING] Starting for {NUM_EPOCHS} epochs...\n")
    loss_history = train(model, names, char2idx, vocab_size, NUM_EPOCHS, LEARNING_RATE)
    print(f"\n[TRAINING] Done. Final avg loss: {loss_history[-1]:.4f}")

    # ── Save model ─────────────────────────────────────────────────────────
    torch.save({
        "model_state" : model.state_dict(),
        "char2idx"    : char2idx,
        "idx2char"    : idx2char,
        "vocab_size"  : vocab_size,
        "hidden_size" : HIDDEN_SIZE,
        "attn_size"   : ATTN_SIZE,
    }, "attention_rnn_model.pt")
    print("[SAVED] Model saved to 'attention_rnn_model.pt'")

    # ── Generate names ─────────────────────────────────────────────────────
    print("\n[GENERATE] Generating 200 names with temperature=0.8 ...\n")
    generated_names = generate_names_batch(
        model, char2idx, idx2char, vocab_size, n=200
    )

    print("Sample generated names:")
    for i, name in enumerate(generated_names[:20], 1):
        print(f"  {i:>2}. {name}")

    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "generated_names_attention_rnn.txt")
    with open(output_path, "w") as f:
        for name in generated_names:
            f.write(name + "\n")
    print(f"\n[SAVED] {len(generated_names)} generated names saved to '{output_path}'")