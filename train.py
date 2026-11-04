"""
train.py
────────
Training e valutazione del modello Redemption utilizzando il nuovo
DataLoader custom basato su Polars e padding delle sequenze.

Flusso:
  1. Carica i file CSV dalla cartella specificata e fa lo split train/val.
  2. Inizializza RedemptionDataLoader (applica padding dinamico).
  3. Modello lineare nn.Linear(6 -> 1) allenato con BCEWithLogitsLoss.
  4. Valutazione: loss, accuracy, precision, recall, F1 + confusion matrix.
  5. Salva i pesi del checkpoint con l'F1 score migliore.
"""

import os
import torch
import torch.nn as nn

# Import dai tuoi nuovi script
from dataset import RedemptionDataLoader
from redemption import Redemption

# ══════════════════════════════════════════════════════════════════
#  CONFIGURAZIONE
# ══════════════════════════════════════════════════════════════════

DATA_DIR = "/home/ale/home-old/unipd/2.2/Progetto-Cyber/Progetto-Cybersecurity/data/output"  # Sostituisci con la cartella contenente i tuoi .csv
SAVE_PATH = "redemption_best.pt"

# Parametri Finestra (da passare al Dataloader Polars)
STRATEGY = "time-windows"  # "n-operations", "time-windows" o "state-windows" (CSV con WindowStart)
WINDOW_SIZE = 5  # Valido se strategy="n-operations"
TIME_WINDOW_SECS = 10.0  # Valido se strategy="time-windows"

# Iperparametri Training
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 64
VAL_RATIO = 0.2
SEED = 42


# ══════════════════════════════════════════════════════════════════
#  TRAIN / EVAL LOOPS
# ══════════════════════════════════════════════════════════════════


def train_epoch(
    model: Redemption,
    loader: RedemptionDataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Esegue un'epoca di training e restituisce la loss media."""
    model.train()
    total_loss = 0.0

    for features, labels in loader:
        features = features.to(device)  # Forma: (Batch, Length, 6)
        labels = labels.to(device)  # Forma: (Batch, Length)

        optimizer.zero_grad()

        # Il modello restituisce (Batch, Length, 1). squeeze(-1) lo fa diventare (Batch, Length)
        logits = model(features).squeeze(-1)

        # Appiattiamo i tensori a (Batch * Length) per calcolare la loss su tutti gli eventi
        loss = criterion(logits.view(-1), labels.view(-1))

        loss.backward()
        optimizer.step()

        # Moltiplichiamo per la dimensione reale del batch per una media corretta
        total_loss += loss.item() * features.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: Redemption,
    loader: RedemptionDataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Valuta il modello e restituisce metriche chiave su tutte le predizioni appiattite."""
    model.eval()
    total_loss = 0.0
    tp = fp = tn = fn = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features).squeeze(-1)
        loss = criterion(logits.view(-1), labels.view(-1))
        total_loss += loss.item() * features.size(0)

        # Calcolo delle predizioni. Appiattiamo anche qui per le metriche
        preds = (torch.sigmoid(logits.view(-1)) >= threshold).float()
        lbls = labels.view(-1)

        tp += ((preds == 1) & (lbls == 1)).sum().item()
        fp += ((preds == 1) & (lbls == 0)).sum().item()
        tn += ((preds == 0) & (lbls == 0)).sum().item()
        fn += ((preds == 0) & (lbls == 1)).sum().item()

    n = len(loader.dataset)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "loss": total_loss / n if n > 0 else 0.0,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def print_metrics(split: str, m: dict):
    print(
        f"  {split:<6} | loss={m['loss']:.4f}"
        f"  acc={m['accuracy']:.4f}"
        f"  prec={m['precision']:.4f}"
        f"  rec={m['recall']:.4f}"
        f"  F1={m['f1']:.4f}"
    )


# ══════════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in uso: {device}\n")

    # ── 1. Data Loading ───────────────────────────────────────────
    if not os.path.exists(DATA_DIR):
        print(
            f"Cartella dati non trovata: {DATA_DIR}. Aggiorna il percorso in CONFIGURAZIONE."
        )
        exit(1)

    print(f"Creazione dei DataLoader da {DATA_DIR}...")
    train_loader, val_loader = RedemptionDataLoader.create_splits_from_folder(
        data_dir=DATA_DIR,
        val_ratio=VAL_RATIO,
        batch_size=BATCH_SIZE,
        seed=SEED,
        strategy=STRATEGY,
        window_size=WINDOW_SIZE,
        time_window_secs=TIME_WINDOW_SECS,
    )

    print(f"  Dataset Train: {len(train_loader.dataset)} finestre")
    print(f"  Dataset Val:   {len(val_loader.dataset)} finestre\n")

    # ── 2. Modello, Loss e Ottimizzatore ──────────────────────────
    model = Redemption().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    print(f"Training Config: {EPOCHS} epoche | strategy={STRATEGY}\n")
    print(f"{'Epoch':>5}  {'TrainLoss':>9}  {'ValLoss':>7}  {'ValF1':>6}  {'LR':>8}")
    print("─" * 50)

    best_f1 = -1.0

    # ── 3. Loop di Addestramento ──────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:>5}  {train_loss:.4f}     "
            f"{val_metrics['loss']:.4f}  "
            f"{val_metrics['f1']:.4f}  "
            f"{lr_now:.1e}"
        )

        # Aggiorna il learning rate in base all'F1 Score di validazione
        scheduler.step(val_metrics["f1"])

        # Salva se migliora F1
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), SAVE_PATH)

    # ── 4. Valutazione Finale ─────────────────────────────────────
    print(f"\n── Valutazione finale (miglior F1 val: {best_f1:.4f}) ──")

    # Ricarica i pesi migliori
    if os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))

    m_train = evaluate(model, train_loader, criterion, device)
    m_val = evaluate(model, val_loader, criterion, device)

    print_metrics("Train", m_train)
    print_metrics("Val", m_val)

    print(
        f"\n  Matrice Risultati (val):\n"
        f"             Negative  Positive\n"
        f"  False  {m_val['fn']:>8}  {m_val['fp']:>8}\n"
        f"  True   {m_val['tn']:>8}  {m_val['tp']:>8}\n"
    )
    print(f"Pesi salvati in: {SAVE_PATH}")
