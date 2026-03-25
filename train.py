"""
train.py
────────
Training e valutazione di Redemption su tutti i PID del log IRP.

Flusso:
  1. Carica e labela il log
  2. Split train/val a livello di PID (no data-leakage)
  3. Pre-estrae le feature con RedemptionDataset
  4. Allena nn.Linear(6,1) con BCEWithLogitsLoss
  5. Valuta su validation set: loss, accuracy, precision, recall, F1
  6. Salva i pesi migliori

Prerequisiti:
  pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

import importlib
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ps = importlib.import_module("parser")
from malice_score import Redemption
from redemption_dataset import split_pid_rows, make_dataloaders


# ══════════════════════════════════════════════════════════════════
#  CONFIGURAZIONE
# ══════════════════════════════════════════════════════════════════

LOG_PATH     = "data/IRP/good_out.log"
LABELED_PATH = "data/IRP/labeled_irp.csv"
SAVE_PATH    = "redemption_best.pt"
RANSOMWARE_PID = "160"           # fallback se labeled_irp.csv non esiste

EPOCHS     = 100
LR         = 1e-3
BATCH_SIZE = 256
VAL_RATIO  = 0.2
SEED       = 42


# ══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_labeled(log_path: str, labeled_path: str):
    """
    Carica il log IRP con le label is_ransomware già per riga.

    Strategia:
      - Se labeled_irp.csv esiste, lo leggiamo direttamente: contiene già
        tutte le colonne parsed + is_ransomware per ogni evento, quindi non
        serve nessun merge (che perdeva le label perché operava per PID).
      - Fallback: parsiamo il log grezzo e assegniamo la label in base al
        RANSOMWARE_PID noto (utile durante lo sviluppo senza CSV labeled).

    Il CSV labeled ha il formato prodotto da label_irp.py:
      type, time, pid, major, …, is_ransomware
    """
    import pandas as pd

    if os.path.exists(labeled_path):
        # Carichiamo direttamente il CSV che ha già is_ransomware per riga
        df = pd.read_csv(labeled_path)
        df['pid'] = df['pid'].astype(str)
        df['time'] = pd.to_datetime(df['time'], errors='coerce')

        # add_dt: ricalcola dt per PID in ordine temporale
        df = df.sort_values(['pid', 'time'])
        df['dt'] = df.groupby('pid')['time'].diff().dt.total_seconds().fillna(0.0)

        df['is_ransomware'] = df['is_ransomware'].fillna(0).astype(int)
        print(f"[load_labeled] caricato {labeled_path}  ({len(df)} righe)")
    else:
        # Fallback: parsare il log grezzo e label da PID noto
        df = ps.load_irp_log(log_path)
        df = ps.add_dt(df)
        df['is_ransomware'] = (df['pid'].astype(str) == RANSOMWARE_PID).astype(int)
        n = df['is_ransomware'].sum()
        print(
            f"[load_labeled] WARN: {labeled_path} non trovato. "
            f"Parsato {log_path}, label assegnata al PID={RANSOMWARE_PID} ({n} righe)."
        )

    n_r = df['is_ransomware'].sum()
    n_b = len(df) - n_r
    print(f"  ransomware={n_r}  benign={n_b}  PID unici={df['pid'].nunique()}")
    return df


def df_to_pid_rows(df) -> dict:
    """
    Converte il DataFrame in dict {pid: list[dict]}.
    to_dict('records') produce una lista di {col: val} per ogni riga.
    """
    return {
        pid: group.to_dict('records')
        for pid, group in df.groupby('pid')
    }


# ══════════════════════════════════════════════════════════════════
#  TRAIN / EVAL
# ══════════════════════════════════════════════════════════════════

def train_epoch(
    model: Redemption,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Esegue un'epoca di training. Restituisce la loss media."""
    model.train()
    total_loss = 0.0

    for features, labels in loader:
        features = features.to(device)   # (B, 6)
        labels   = labels.to(device)     # (B,)

        optimizer.zero_grad()
        logits = model(features).squeeze(-1)   # (B,)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: Redemption,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Valuta il modello sul loader dato.

    Returns:
        dict con loss, accuracy, precision, recall, f1
    """
    model.eval()
    total_loss = 0.0

    # accumulatori per confusion matrix
    tp = fp = tn = fn = 0

    for features, labels in loader:
        features = features.to(device)
        labels   = labels.to(device)

        logits = model(features).squeeze(-1)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * len(labels)

        preds = (torch.sigmoid(logits) >= threshold).float()

        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    n = len(loader.dataset)
    accuracy  = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    return {
        'loss':      total_loss / n,
        'accuracy':  accuracy,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


def print_metrics(split: str, metrics: dict):
    print(
        f"  {split:<6} | "
        f"loss={metrics['loss']:.4f}  "
        f"acc={metrics['accuracy']:.4f}  "
        f"prec={metrics['precision']:.4f}  "
        f"rec={metrics['recall']:.4f}  "
        f"F1={metrics['f1']:.4f}"
    )


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── 1. Dati ──────────────────────────────────────────────────
    print(f"Carico {LOG_PATH} …")
    df = load_labeled(LOG_PATH, LABELED_PATH)
    print(
        f"  {len(df)} righe | "
        f"{df['pid'].nunique()} PID | "
        f"{df['is_ransomware'].sum()} eventi ransomware\n"
    )

    pid_rows = df_to_pid_rows(df)

    # ── 2. Split train/val a livello di PID ──────────────────────
    train_rows, val_rows = split_pid_rows(pid_rows, val_ratio=VAL_RATIO, seed=SEED)

    # ── 3. Dataset e DataLoader ──────────────────────────────────
    print()
    train_loader, val_loader = make_dataloaders(
        train_rows, val_rows, batch_size=BATCH_SIZE
    )

    # ── 4. Modello, loss, ottimizzatore ──────────────────────────
    model     = Redemption().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    print(f"\nModello: {model}")
    print(f"Training: {EPOCHS} epoche | lr={LR} | batch={BATCH_SIZE}\n")
    print(f"{'Epoch':>5}  {'TrainLoss':>9}  {'ValLoss':>7}  {'F1':>6}  {'LR':>10}")
    print("─" * 55)

    best_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss  = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        lr_now = optimizer.param_groups[0]['lr']
        print(
            f"{epoch:>5}  "
            f"{train_loss:.4f}     "
            f"{val_metrics['loss']:.4f}  "
            f"{val_metrics['f1']:.4f}  "
            f"{lr_now:.2e}"
        )

        # Scheduler basato su F1 validation
        scheduler.step(val_metrics['f1'])

        # Salva il modello migliore per F1
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), SAVE_PATH)

    # ── 5. Valutazione finale ────────────────────────────────────
    print(f"\n── Valutazione finale (miglior F1 su val: {best_f1:.4f}) ──")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))

    train_metrics = evaluate(model, train_loader, criterion, device, threshold=0.12)
    val_metrics   = evaluate(model, val_loader,   criterion, device, threshold=0.12)

    print_metrics("Train", train_metrics)
    print_metrics("Val",   val_metrics)

    print(
        f"\n  Confusion matrix (val):\n"
        f"              Pred 0    Pred 1\n"
        f"  Actual 0  {val_metrics['tn']:>8}  {val_metrics['fp']:>8}\n"
        f"  Actual 1  {val_metrics['fn']:>8}  {val_metrics['tp']:>8}\n"
    )
    print(f"Pesi salvati in {SAVE_PATH}")