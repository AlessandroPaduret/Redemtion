"""
redemption_dataset.py
─────────────────────
Dataset PyTorch per Redemption con sliding window a livello di processo.

Perché sliding window
───────────────────────────────────────
Redemption classifica processi, non singole operazioni IRP.  Un processo può
iniziare come benign e diventare ransomware a metà esecuzione (es. code
injection).  Etichettare ogni IRP individualmente sarebbe sbagliato: l'evento
"leggi file" ha la stessa forma prima e dopo l'injection.

Approccio corretto:
  - Finestra scorrevole di W eventi su ogni PID.
  - Le feature vengono estratte in ordine temporale su tutti i W eventi;
    lo stato di Redemption (entropia letta, file scritti, ecc.) si accumula
    per l'intera finestra → il vettore dell'ULTIMO evento encode l'intera storia.
  - La label della finestra è quella dell'ULTIMO evento: cattura il momento
    in cui il processo diventa malevolo senza "contaminare" le finestre
    precedenti ancora benign.
  - Il gradiente scorre UNA SOLA VOLTA per finestra (non per ogni IRP).

Struttura del campione:
  features : Tensor (6,)   — feature dell'ultimo evento della finestra
  label    : Tensor scalar — is_ransomware dell'ultimo evento

Split:
  A livello di PID per evitare data-leakage (tutti gli eventi di un processo
  stanno interamente in train o in val, mai in entrambi).
"""

from __future__ import annotations

import importlib
import random
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

ps = importlib.import_module("parser")
from redemption import Redemption


# ══════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════

class RedemptionDataset(Dataset):
    """
    Dataset flat di finestre scorrevoli.

    Per ogni PID con almeno W eventi, si generano sliding windows di ampiezza W
    con passo STRIDE.  Per ogni finestra si processa l'intera sequenza in ordine
    (stato Redemption resettato all'inizio di ogni finestra) e si conserva solo
    il vettore di feature dell'ultimo evento come rappresentazione della finestra.

    Args:
        pid_rows:   dict {pid: list[dict]}  — output di df_to_pid_rows()
        window_size: W — numero di eventi per finestra
        stride:     passo tra finestre consecutive (default = W/2, 50% overlap)
        model:      istanza Redemption per feature extraction (pesi non modificati)
        verbose:    stampa statistiche dopo la costruzione
    """

    def __init__(
        self,
        pid_rows: dict,
        window_size: int = 50,
        stride: Optional[int] = None,
        model: Optional[Redemption] = None,
        verbose: bool = True,
    ):
        if model is None:
            model = Redemption()
        if stride is None:
            stride = max(1, window_size // 2)

        all_features: list[torch.Tensor] = []
        all_labels:   list[float]        = []

        n_ransomware = 0
        n_benign     = 0
        skipped      = 0

        for pid, rows in pid_rows.items():
            if len(rows) < window_size:
                skipped += 1
                continue

            # Scorriamo la sequenza del PID con la sliding window
            for start in range(0, len(rows) - window_size + 1, stride):
                window = rows[start : start + window_size]

                # Resettiamo lo stato all'inizio di ogni finestra: ogni finestra
                # è indipendente. process_pid processa i W eventi in ordine.
                feat_seq = model.process_pid(window, reset=True)  # (W, 6)

                # Prendiamo SOLO il vettore dell'ultimo passo: encode tutto
                # l'accumulo dello stato visto nella finestra.
                last_feat = feat_seq[-1]   # (6,)

                # Label dell'ultimo evento: se il processo è diventato
                # ransomware entro la fine della finestra, lo vediamo qui.
                label = float(window[-1].get('is_ransomware', 0))

                all_features.append(last_feat)
                all_labels.append(label)

                if label == 1.0:
                    n_ransomware += 1
                else:
                    n_benign += 1

        if not all_features:
            raise ValueError(
                f"Nessuna finestra generata. "
                f"Tutti i PID hanno meno di {window_size} eventi?"
            )

        self.features = torch.stack(all_features)          # (N_windows, 6)
        self.labels   = torch.tensor(all_labels)           # (N_windows,)
        self.window_size = window_size
        self.stride      = stride

        if verbose:
            tot = len(self.labels)
            print(
                f"[RedemptionDataset] {tot} finestre"
                f" (W={window_size}, stride={stride})"
                f" | ransomware={n_ransomware} ({100*n_ransomware/tot:.1f}%)"
                f" | benign={n_benign} ({100*n_benign/tot:.1f}%)"
                f" | PID skippati (troppo corti)={skipped}"
            )

    # ── Dataset protocol ────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Restituisce (feature_vector, label) per la finestra all'indice idx.
        feature_vector: (6,) — vettore dell'ultimo evento della finestra
        label:          scalar float — 0.0 o 1.0
        """
        return self.features[idx], self.labels[idx]


# ══════════════════════════════════════════════════════════════════
#  SPLIT A LIVELLO DI PID
# ══════════════════════════════════════════════════════════════════

def split_pid_rows(
    pid_rows: dict,
    val_ratio: float = 0.2,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[dict, dict]:
    """
    Divide pid_rows in train e validation mantenendo interi i gruppi per PID.

    Splitting a livello di PID (non di evento) per evitare data-leakage:
    eventi dello stesso processo non devono apparire in entrambi i set.

    Args:
        pid_rows:  dict {pid: list[dict]}
        val_ratio: frazione di PID da usare come validation
        seed:      seme per riproducibilità

    Returns:
        train_rows, val_rows
    """
    rng  = random.Random(seed)
    pids = list(pid_rows.keys())
    rng.shuffle(pids)

    n_val      = max(1, int(len(pids) * val_ratio))
    val_pids   = set(pids[:n_val])
    train_pids = set(pids[n_val:])

    train_rows = {p: pid_rows[p] for p in train_pids}
    val_rows   = {p: pid_rows[p] for p in val_pids}

    if verbose:
        print(
            f"[split_pid_rows] train={len(train_rows)} PID"
            f" | val={len(val_rows)} PID"
            f" (val_ratio={val_ratio})"
        )
    return train_rows, val_rows


# ══════════════════════════════════════════════════════════════════
#  FACTORY DataLoader
# ══════════════════════════════════════════════════════════════════

def make_dataloaders(
    train_rows: dict,
    val_rows:   dict,
    window_size: int = 50,
    stride: Optional[int] = None,
    batch_size: int = 256,
    num_workers: int = 0,
    verbose: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Crea i DataLoader di train e validation.

    Un'unica istanza di Redemption (pesi non allenati) viene usata
    solo per la feature extraction offline.

    Args:
        train_rows, val_rows: dict {pid: list[dict]}
        window_size:          W eventi per finestra
        stride:               passo sliding window (default W/2)
        batch_size:           campioni per batch durante il training
        num_workers:          worker paralleli

    Returns:
        train_loader, val_loader
    """
    extractor = Redemption()   # solo per feature extraction, non allenato qui

    train_ds = RedemptionDataset(
        train_rows, window_size=window_size, stride=stride,
        model=extractor, verbose=True,
    )
    val_ds = RedemptionDataset(
        val_rows, window_size=window_size, stride=stride,
        model=extractor, verbose=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if verbose:
        print(
            f"[make_dataloaders] train={len(train_ds)} finestre"
            f" | val={len(val_ds)} finestre"
            f" | batch_size={batch_size}"
        )
    return train_loader, val_loader