import re
import json
from pathlib import Path
from collections import Counter

import re
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
# Warn about NumPy version
_major = int(np.__version__.split('.', 1)[0])
if _major >= 2:
    import warnings
    warnings.warn(
        f"Detected NumPy v{np.__version__}, which may be incompatible with some compiled modules."
        " Consider installing numpy<2 for full compatibility."
    )
_major = int(np.__version__.split('.', 1)[0])
if _major >= 2:
    raise RuntimeError(
        f"Detected NumPy v{np.__version__}, which is incompatible with this module. "
        "Please install numpy<2 (e.g., `pip install 'numpy<2'`)."
    )

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except RuntimeError as e:
    msg = str(e)
    if 'compiled using NumPy' in msg:
        raise RuntimeError(
            "PyTorch could not import because of a NumPy version mismatch. "
            "Please install a NumPy version <2 (e.g., `pip install 'numpy<2'`) and retry."
        )
    else:
        raise

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import torch.optim as optim

# Detect compute device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Text Preprocessing ----------
def clean_text_series(texts: pd.Series) -> pd.Series:
    """
    Lowercase, strip URLs/punctuation, collapse whitespace.
    """
    def clean_one(s: str) -> str:
        s = s.lower()
        s = re.sub(r"https?://\S+|www\.\S+", "", s)
        s = re.sub(r"[^a-z\s]", "", s)
        return re.sub(r"\s+", " ", s).strip()
    return texts.map(clean_one)


def filter_texts_by_length(texts: pd.Series, min_len: int = 3, max_len: int = 60) -> pd.Series:
    """
    Keep only texts whose token count is between min_len and max_len.
    """
    lengths = texts.str.split().apply(len)
    return texts[(lengths >= min_len) & (lengths <= max_len)]

# ---------- Vocabulary & Sequencing ----------
def build_token_index(corpus: list[str], max_tokens: int = 10000) -> dict:
    """
    Build word→index dict with special tokens PAD=0, OOV=1.
    """
    counter = Counter()
    for sent in corpus:
        counter.update(sent.split())
    common = counter.most_common(max_tokens - 2)
    token_idx = {tok: idx + 2 for idx, (tok, _) in enumerate(common)}
    token_idx['<PAD>'] = 0
    token_idx['<OOV>'] = 1
    return token_idx


def texts_to_tensor(corpus: list[str], token_idx: dict, seq_length: int = 200) -> torch.Tensor:
    """
    Convert list of texts into padded/truncated index tensors.
    """
    pad_id = token_idx['<PAD>']
    oov_id = token_idx['<OOV>']
    sequences = []
    for sent in corpus:
        ids = [token_idx.get(w, oov_id) for w in sent.split()]
        if len(ids) < seq_length:
            ids += [pad_id] * (seq_length - len(ids))
        else:
            ids = ids[:seq_length]
        sequences.append(ids)
    return torch.tensor(sequences, dtype=torch.long)

# ---------- Dataset & Model ----------
class TweetsDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor | None = None):
        self.features = features
        self.labels   = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        x = self.features[idx]
        if self.labels is None:
            return x
        return x, self.labels[idx]


class TweetEmotionGRU(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_classes: int, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=0.5)
        self.fc   = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x)
        _, h = self.gru(e)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.drop(h))

# ---------- Training & Evaluation ----------
def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn, opt) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        preds = model(xb)
        loss  = loss_fn(preds, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def validate(model: nn.Module, loader: DataLoader, loss_fn) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            total_loss += loss_fn(out, yb).item() * xb.size(0)
            all_preds.extend(out.argmax(dim=1).cpu().tolist())
            all_true.extend(yb.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_true, all_preds, average='macro')
    return avg_loss, f1

# ---------- Plotting ----------
def plot_history(hist: dict[str, list[float]]):
    epochs = range(1, len(hist['train_loss']) + 1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, hist['train_loss'], label='Train Loss')
    plt.plot(epochs, hist['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss Curves')

    plt.subplot(1,2,2)
    plt.plot(epochs, hist['val_f1'], label='Val F1')
    plt.xlabel('Epoch'); plt.ylabel('F1'); plt.legend(); plt.title('F1 Score')
    plt.tight_layout()
    plt.show()

# ---------- Main Pipeline ----------
def main():
    # Load data
    train = pd.read_csv('train_kaggle.csv')
    test  = pd.read_csv('test_kaggle.csv')

    # Preprocess
    train['text'] = clean_text_series(train['text'])
    train         = train.dropna(subset=['text'])
    train['text'] = filter_texts_by_length(train['text'])
    test['text']  = clean_text_series(test['text'])

    # Encode labels
    le     = LabelEncoder()
    y_all  = le.fit_transform(train['label'])
    n_cl   = len(le.classes_)

    # Vocab & sequences
    vocab     = build_token_index(train['text'].tolist())
    X_all     = texts_to_tensor(train['text'].tolist(), vocab)
    X_test    = texts_to_tensor(test['text'].tolist(), vocab)

    # Train/Val split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    tr_ds = TweetsDataset(X_tr, torch.tensor(y_tr))
    va_ds = TweetsDataset(X_va, torch.tensor(y_va))
    tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=64)
    test_dl = DataLoader(TweetsDataset(X_test), batch_size=64)

    # Model setup
    pad_id    = vocab['<PAD>']
    model     = TweetEmotionGRU(
        vocab_size=len(vocab), emb_dim=300,
        hidden_dim=256, num_classes=n_cl, pad_id=pad_id
    ).to(DEVICE)

    class_wts = compute_class_weight(
        class_weight='balanced', classes=np.unique(y_tr), y=y_tr
    )
    wts       = torch.tensor(class_wts, dtype=torch.float).to(DEVICE)
    loss_fn   = nn.CrossEntropyLoss(weight=wts)
    optimizer = optim.Adam(model.parameters(), lr=4e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    # Training loop w/ early stop
    best_f1 = 0.0
    patience= 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    for e in range(1, 31):
        tl = train_one_epoch(model, tr_dl, loss_fn, optimizer)
        vl, vf = validate(model, va_dl, loss_fn)
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['val_f1'].append(vf)

        print(f"Epoch {e}: Train Loss {tl:.4f}, Val Loss {vl:.4f}, Val F1 {vf:.4f}")
        scheduler.step(vf)

        if vf > best_f1:
            best_f1 = vf
            patience = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience += 1
            if patience >= 4:
                print("Early stopping at epoch", e)
                break

    # Plot metrics
    plot_history(history)

    # Inference on test set
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb in test_dl:
            xb = xb.to(DEVICE)
            logits = model(xb)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())

    # Save submission
    submission = pd.DataFrame({'ID': test['ID'], 'label': le.inverse_transform(all_preds)})
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

    # Persist artifacts
    torch.save(model.state_dict(), 'best_model.pt')
    with open('vocab.json','w') as f:
        json.dump(vocab, f)
    with open('label_encoder.pkl','wb') as f:
        torch.save(le, f)

if __name__ == '__main__':
    main()



