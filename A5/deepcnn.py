# deep_cnn.py
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------------------
#  Config
# --------------------
DATA_PATH = Path("data_proc") / "grouped_chunks.csv"
SEED = 42
N_SPLITS = 3          # k-folds (can change to 3 if you prefer)
BATCH_SIZE = 16       # as per paper
N_EPOCHS = 10         # as per paper
EMBED_DIM = 8         # paper
NUM_FILTERS = 64      # paper
KERNEL_SIZE = 8       # paper
HIDDEN_DIM = 16       # paper
MAX_SEQ_LEN_CAP = 2000  # safety cap on sequence length

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------
#  Utils
# --------------------

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_vocab(texts, min_freq: int = 1):
    """
    Build vocabulary from a list of "OP1 OP2 ..." strings.
    Returns: token_to_id dict with PAD=0, UNK=1 reserved.
    """
    counter = Counter()
    for t in texts:
        counter.update(t.split())

    token_to_id = {"<PAD>": 0, "<UNK>": 1}
    for tok, freq in counter.items():
        if freq >= min_freq:
            token_to_id[tok] = len(token_to_id)

    return token_to_id


def texts_to_sequences(texts, token_to_id, max_len):
    """
    Convert a list of opcode strings to padded/truncated integer sequences.
    """
    pad_id = token_to_id["<PAD>"]
    unk_id = token_to_id["<UNK>"]

    seqs = []
    for t in texts:
        tokens = t.split()
        ids = [token_to_id.get(tok, unk_id) for tok in tokens]
        # truncate
        ids = ids[:max_len]
        # pad
        if len(ids) < max_len:
            ids.extend([pad_id] * (max_len - len(ids)))
        seqs.append(ids)

    return np.array(seqs, dtype=np.int64)


# --------------------
#  Dataset + Model
# --------------------

class OpcodeDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.as_tensor(sequences, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class CNNMalwareDetector(nn.Module):
    """
    Deep Android Malware Detection architecture (slightly generalized):

      Embedding (|V| x EMBED_DIM)
      -> 1D Conv (EMBED_DIM -> NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE//2)
      -> ReLU
      -> Global max pool over time
      -> FC hidden (NUM_FILTERS -> HIDDEN_DIM) + ReLU
      -> FC output (HIDDEN_DIM -> num_classes)
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = EMBED_DIM,
        num_filters: int = NUM_FILTERS,
        kernel_size: int = KERNEL_SIZE,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(num_filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch, seq_len]
        emb = self.embedding(x)          # [batch, seq_len, embed_dim]
        emb = emb.permute(0, 2, 1)       # [batch, embed_dim, seq_len]
        conv_out = self.relu(self.conv(emb))  # [batch, num_filters, seq_len]
        pooled, _ = torch.max(conv_out, dim=2)  # [batch, num_filters]
        h = self.relu(self.fc1(pooled))        # [batch, hidden_dim]
        logits = self.fc2(h)                   # [batch, num_classes]
        return logits


# --------------------
#  Training / Eval loops
# --------------------

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)

    return running_loss / len(loader.dataset)


def eval_model(model, loader):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch_y.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    preds = np.argmax(all_logits, axis=1)
    acc = accuracy_score(all_labels, preds)
    macro_f1 = f1_score(all_labels, preds, average="macro")

    return acc, macro_f1, all_labels, preds


# --------------------
#  K-fold experiment
# --------------------

def run_cnn_kfold():
    set_seed(SEED)

    df = pd.read_csv(DATA_PATH)
    print(df.head())

    # Encode labels to integers
    df["label"] = df["label"].astype("category")
    label_to_idx = {cat: i for i, cat in enumerate(df["label"].cat.categories)}
    idx_to_label = {i: cat for cat, i in label_to_idx.items()}
    y = df["label"].map(label_to_idx).values
    X_text = df["text"].values
    groups = df["group"].values

    num_classes = len(label_to_idx)
    print(f"Classes: {label_to_idx}")

    # Group-aware stratified K-fold (same idea as your ml.py)
    sgkf = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SEED
    )

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_text, y, groups), start=1):
        print(f"\n========== Fold {fold}/{N_SPLITS} ==========")
        X_train_text = X_text[train_idx]
        y_train = y[train_idx]
        X_test_text = X_text[test_idx]
        y_test = y[test_idx]

        # Build vocab from training data only
        token_to_id = build_vocab(X_train_text, min_freq=1)
        vocab_size = len(token_to_id)
        print(f"Vocab size (train only): {vocab_size}")

        # Choose a max sequence length based on train set (cap to avoid crazy long sequences)
        train_lengths = [len(t.split()) for t in X_train_text]
        max_len = min(int(np.percentile(train_lengths, 95)), MAX_SEQ_LEN_CAP)
        max_len = max(max_len, 8)  # at least kernel size
        print(f"Max sequence length (95th percentile, capped): {max_len}")

        # Encode texts
        X_train_seq = texts_to_sequences(X_train_text, token_to_id, max_len)
        X_test_seq = texts_to_sequences(X_test_text, token_to_id, max_len)

        # Build datasets/loaders
        train_ds = OpcodeDataset(X_train_seq, y_train)
        test_ds = OpcodeDataset(X_test_seq, y_test)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
        )

        # Instantiate model
        model = CNNMalwareDetector(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embed_dim=EMBED_DIM,
            num_filters=NUM_FILTERS,
            kernel_size=KERNEL_SIZE,
            hidden_dim=HIDDEN_DIM,
        ).to(device)

        # Class weights for imbalance (inverse frequency)
        class_counts = np.bincount(y_train, minlength=num_classes)
        class_weights = class_counts.sum() / (len(class_counts) * class_counts)
        class_weights = torch.as_tensor(class_weights, dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)

        # Train
        for epoch in range(1, N_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            acc, macro_f1, _, _ = eval_model(model, test_loader)
            print(
                f"Fold {fold} Epoch {epoch:02d} "
                f"- TrainLoss: {train_loss:.4f}, "
                f"Test Acc: {acc:.4f}, Test Macro-F1: {macro_f1:.4f}"
            )

        # Final evaluation for this fold
        acc, macro_f1, y_true, y_pred = eval_model(model, test_loader)
        print("\nClassification report (fold {}):".format(fold))
        # Determine which labels exist in this fold's test set
        present_labels = sorted(set(y_true) | set(y_pred))

        print(classification_report(
            y_true,
            y_pred,
            labels=present_labels,
            target_names=[idx_to_label[i] for i in present_labels]
        ))


        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))

        fold_metrics.append((acc, macro_f1))

    # Summary across folds
    accs = [m[0] for m in fold_metrics]
    f1s = [m[1] for m in fold_metrics]
    print("\n========== CNN Summary across folds ==========")
    print(f"Mean Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"Mean Macro-F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")


# --------------------
#  Main
# --------------------

if __name__ == "__main__":
    run_cnn_kfold()
