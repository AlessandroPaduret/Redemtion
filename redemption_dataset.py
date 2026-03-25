"""
redemption_dataset.py
─────────────────────
Dataset PyTorch per il modello Redemption.

Strategia:
  Redemption è stateful: le feature di ogni evento dipendono dagli eventi
  precedenti dello stesso PID (entropia letta, file scritti, ecc.).
  Non possiamo mescolare eventi di PID diversi in un batch senza prima
  estrarre le feature in ordine temporale.

  Approccio in due fasi:
    1. PRE-ESTRAZIONE (offline, in __init__)
       Per ogni PID, si processa l'intera sequenza in ordine e si ottengono
       i vettori di feature (N_pid, 6).  Lo stato viene resettato tra un PID
       e l'altro.

    2. TRAINING (online, in __getitem__)
       Il dataset è piatto: ogni campione è una singola coppia (feature, label).
       I DataLoader standard funzionano senza vincoli.

Suddivisione train/val:
  La split avviene a livello di PID, non di singolo evento, per evitare
  data-leakage (eventi dello stesso processo non devono finire in entrambi
  i set).
"""

from __future__ import annotations

import importlib
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Importiamo il nostro parser.py senza collidere col built-in 'parser'
ps = importlib.import_module("parser")
from malice_score import Redemption


# ══════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════

class RedemptionDataset(Dataset):
    """
    Dataset flat: ogni campione è un singolo evento IRP con le sue 6 feature
    e la label binaria (0 = benign, 1 = ransomware).

    Args:
        pid_rows:   dict {pid: list[dict]}  — output di df_to_pid_rows()
        model:      istanza di Redemption usata solo per l'estrazione feature;
                    i pesi del Linear non vengono toccati qui.
        verbose:    se True, stampa un resoconto dopo la pre-estrazione.
    """

    def __init__(
        self,
        pid_rows: dict,
        model: Optional[Redemption] = None,
        verbose: bool = True,
    ):
        if model is None:
            model = Redemption()

        all_features: list[torch.Tensor] = []
        all_labels:   list[torch.Tensor] = []

        n_ransomware = 0
        n_benign     = 0
        skipped      = 0

        for pid, rows in pid_rows.items():
            if not rows:
                skipped += 1
                continue

            # Pre-estrazione: stato resettato automaticamente per ogni PID
            features = model.process_pid(rows, reset=True)   # (N, 6)

            labels = torch.tensor(
                [float(r.get('is_ransomware', 0)) for r in rows],
                dtype=torch.float32,
            )                                                 # (N,)

            all_features.append(features)
            all_labels.append(labels)

            n_ransomware += int(labels.sum().item())
            n_benign     += int((labels == 0).sum().item())

        if not all_features:
            raise ValueError("Nessun PID valido trovato nel dataset.")

        # Tensori piatti: (tot_eventi, 6) e (tot_eventi,)
        self.features = torch.cat(all_features, dim=0)
        self.labels   = torch.cat(all_labels,   dim=0)

        if verbose:
            tot = len(self.labels)
            print(
                f"[RedemptionDataset] {tot} eventi totali"
                f" | ransomware={n_ransomware} ({100*n_ransomware/tot:.1f}%)"
                f" | benign={n_benign} ({100*n_benign/tot:.1f}%)"
                f" | PID skippati={skipped}"
            )

    # ── Dataset protocol ────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Restituisce (feature_vector, label) per l'evento all'indice idx."""
        return self.features[idx], self.labels[idx]


# ══════════════════════════════════════════════════════════════════
#  SPLIT TRAIN / VAL a livello di PID
# ══════════════════════════════════════════════════════════════════

def split_pid_rows(
    pid_rows: dict,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[dict, dict]:
    """
    Divide il dizionario pid_rows in train e validation mantenendo interi i
    gruppi per PID (niente data-leakage tra eventi dello stesso processo).

    Args:
        pid_rows:  dict {pid: list[dict]}
        val_ratio: frazione di PID da usare come validation
        seed:      seme per riproducibilità

    Returns:
        train_rows, val_rows  — stessa struttura di pid_rows
    """
    import random
    rng = random.Random(seed)

    pids = list(pid_rows.keys())
    rng.shuffle(pids)

    n_val = max(1, int(len(pids) * val_ratio))
    val_pids   = set(pids[:n_val])
    train_pids = set(pids[n_val:])

    train_rows = {p: pid_rows[p] for p in train_pids}
    val_rows   = {p: pid_rows[p] for p in val_pids}

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
    val_rows: dict,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Crea i DataLoader di train e validation a partire dai pid_rows già splittati.

    Args:
        train_rows, val_rows: dict {pid: list[dict]}
        batch_size:           campioni per batch
        num_workers:          worker paralleli per il DataLoader

    Returns:
        train_loader, val_loader
    """
    shared_model = Redemption()   # usato solo per feature extraction, non allenato qui

    train_ds = RedemptionDataset(train_rows, model=shared_model, verbose=True)
    val_ds   = RedemptionDataset(val_rows,   model=shared_model, verbose=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,          # mischiamo gli eventi (i PID sono già stati processati in ordine)
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

    print(
        f"[make_dataloaders] train={len(train_ds)} eventi"
        f" | val={len(val_ds)} eventi"
        f" | batch_size={batch_size}"
    )
    return train_loader, val_loader