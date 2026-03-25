"""
redemption.py
─────────────
Implementazione allenabile di Redemption (Kharraz et al., 2017) come nn.Module.

Architettura:
  - Feature extractor  : stateful, basato su regole, opera su singoli eventi IRP
  - Classificatore     : nn.Linear(6, 1) allenabile, combina le 6 feature

Le 6 feature (tutte ∈ [0, 1]):
  f1  Entropy Ratio          – aumento di entropia tra lettura e scrittura sullo stesso offset
  f2  File Overwrite         – frazione del contenuto del file sovrascritto
  f3  File Deletion          – richiesta di cancellazione (0/1)
  f4  Sequential Writes      – scritture sequenziali su directory già esplorate
  f5  Cross-class Writes     – scritture su almeno 2 estensioni diverse (0/1)
  f6  Write Frequency        – 1/dt tra eventi consecutivi (normalizzato)

Utilizzo tipico:
  model = Redemption()
  features = model.process_pid(rows)          # (N, 6) — feature extraction
  logits   = model(features)                  # (N, 1) — classificazione
  loss     = nn.BCEWithLogitsLoss()(logits.squeeze(), labels)
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════
#  MODELLO
# ══════════════════════════════════════════════════════════════════

class Redemption(nn.Module):
    """
    Wrapper nn.Module che separa l'estrazione di feature (stateful, basata su
    regole) dalla combinazione lineare allenabile.

    Lo stato interno viene resettato per ogni nuovo PID tramite reset_state().
    """

    N_FEATURES = 6

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(self.N_FEATURES, 1)
        self._reset_state()

    # ── State management ────────────────────────────────────────

    def _reset_state(self):
        """Azzera lo stato accumulato; da chiamare all'inizio di ogni nuovo PID."""

        # f1: ultima entropia di lettura indicizzata da "(dest_path, offset)"
        self._read_entropy: dict[str, float] = {}

        # f2: stato scritture per file
        #   dest_path → {'initial_size': int, 'written': dict[offset, buf_size]}
        #   Usiamo offset come chiave per evitare di contare due volte la stessa zona.
        self._file_write_state: dict[str, dict] = {}

        # f4: directory già esplorate (da DIRECTORY_CONTROL / QUERY_DIRECTORY)
        self._discovered_paths: set[str] = set()
        # f4: file unici scritti per ogni parent directory
        self._path_writes: dict[str, set] = {}

        # f5: estensioni su cui il processo ha scritto
        self._written_extensions: set[str] = set()

        # f6: timestamp dell'ultimo evento (oggetto datetime o float)
        self._last_time: Optional[object] = None

    # ── nn.Module forward ────────────────────────────────────────

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Applica il classificatore lineare alle feature già estratte.

        Args:
            features: Tensor (N, 6)
        Returns:
            logit: Tensor (N, 1)  → usare con BCEWithLogitsLoss
        """
        return self.fc(features)

    # ── Public API ───────────────────────────────────────────────

    def process_pid(self, rows: list[dict], reset: bool = True) -> torch.Tensor:
        """
        Processa una sequenza di eventi IRP per un singolo PID e restituisce
        il tensore di feature pronto per il forward pass.

        Args:
            rows:  lista di dict prodotti da parser.parse_line()
            reset: se True (default), azzera lo stato interno prima di elaborare

        Returns:
            Tensor (len(rows), 6) di float32
        """
        if reset:
            self._reset_state()
        features = [self.extract_features(r) for r in rows]
        return torch.tensor(features, dtype=torch.float32)

    def extract_features(self, r: dict) -> list[float]:
        """
        Estrae le 6 feature da un singolo evento di log e aggiorna lo stato.
        Deve essere chiamato in ordine temporale su eventi dello stesso PID.

        Returns:
            Lista [f1, f2, f3, f4, f5, f6]
        """
        f1 = self._entropy_ratio(r)
        self._update_read_entropy(r)     # aggiorniamo DOPO il calcolo: f1 confronta
                                         # la scrittura corrente con letture passate

        self._update_file_overwrite(r)
        f2 = self._file_overwrite(r)

        f3 = self._file_deletion(r)

        self._update_sequential_writes(r)
        f4 = self._sequential_writes(r)

        self._update_extension_writes(r)
        f5 = self._cross_class_writes()

        f6 = self._write_frequency(r)

        return [f1, f2, f3, f4, f5, f6]

    # ══════════════════════════════════════════════════════════════
    #  FEATURE 1 — Entropy Ratio
    # ══════════════════════════════════════════════════════════════

    def _entropy_ratio(self, r: dict) -> float:
        """
        Confronta l'entropia del buffer scritto con quella dell'ultima lettura
        sullo stesso (dest_path, offset).  Un'entropia più alta in scrittura
        rispetto alla lettura è il segnale classico di cifratura ransomware.

        Formula: 1 - read_entropy / write_entropy  ∈ [0, 1]
        Restituisce 0 se non è una scrittura o se non esiste una lettura precedente
        sullo stesso offset, o se l'entropia non è aumentata.
        """
        if r.get('major') != 'IRP_MJ_WRITE':
            return 0.0
        key = f"{r.get('dest_path')}:{r.get('offset')}"
        read_ent = self._read_entropy.get(key)
        write_ent = r.get('entropy') or 0.0
        if read_ent is not None and write_ent > read_ent and write_ent > 0:
            return 1.0 - read_ent / write_ent
        return 0.0

    def _update_read_entropy(self, r: dict):
        """Memorizza l'entropia dell'ultimo evento di lettura per (dest_path, offset)."""
        if r.get('major') == 'IRP_MJ_READ':
            key = f"{r.get('dest_path')}:{r.get('offset')}"
            self._read_entropy[key] = r.get('entropy') or 0.0

    # ══════════════════════════════════════════════════════════════
    #  FEATURE 2 — File Content Overwrite
    # ══════════════════════════════════════════════════════════════

    def _file_overwrite(self, r: dict) -> float:
        """
        Stima la frazione del file sovrascritta dal processo.

        Formula: Σ buf_size per offset unico / initial_file_size  ∈ [0, 1]
        Un ransomware tipicamente sovrascrive l'intero contenuto prima di
        rinominare il file con l'estensione cifrata.
        """
        path = r.get('dest_path')
        state = self._file_write_state.get(path)
        if state is None or state['initial_size'] == 0:
            return 0.0
        bytes_written = sum(state['written'].values())
        return min(bytes_written / state['initial_size'], 1.0)

    def _update_file_overwrite(self, r: dict):
        """
        Tiene traccia dei byte scritti per file, indicizzati per offset.
        Usa l'offset come chiave così le sovrascritture sullo stesso blocco
        non vengono contate due volte.
        """
        if r.get('major') != 'IRP_MJ_WRITE':
            return
        path = r.get('dest_path')
        if path is None:
            return
        if path not in self._file_write_state:
            self._file_write_state[path] = {
                'initial_size': r.get('presize') or 0,
                'written': {}      # {offset: buf_size}
            }
        offset = r.get('offset') or 0
        buf_size = r.get('buffersize') or 0
        # Sovrascrivi l'entry esistente: stessa zona, aggiornato con ultima size
        self._file_write_state[path]['written'][offset] = buf_size

    # ══════════════════════════════════════════════════════════════
    #  FEATURE 3 — File Deletion
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _file_deletion(r: dict) -> float:
        """
        Rileva richieste esplicite di cancellazione tramite FileDispositionInformation.
        Restituisce 1.0 se il flag 'delete' è True, 0.0 altrimenti.
        """
        return float(
            r.get('major') == 'IRP_MJ_SET_INFORMATION'
            and r.get('fileinfoclass') == 'FileDispositionInformation'
            and r.get('delete') is True
        )

    # ══════════════════════════════════════════════════════════════
    #  FEATURE 4 — Sequential Writes
    # ══════════════════════════════════════════════════════════════

    def _sequential_writes(self, r: dict) -> float:
        """
        Misura quanto il processo scrive in modo esteso su directory già esplorate.

        Formula: 1 - 1 / (1 + |file unici scritti nel parent path|)  ∈ [0, 1)
        Il valore cresce con il numero di file unici toccati nella stessa directory.
        """
        path = r.get('dest_path')
        if not path:
            return 0.0
        parent = _parent_path(path)
        n_files = len(self._path_writes.get(parent, ()))
        return 1.0 - 1.0 / (1.0 + n_files) if n_files else 0.0

    def _update_sequential_writes(self, r: dict):
        """
        Aggiorna lo stato per f4:
        - registra la directory quando viene esplorata (QUERY_DIRECTORY)
        - traccia i file scritti all'interno delle directory esplorate
        """
        major = r.get('major')
        minor = r.get('minor')
        path  = r.get('dest_path')
        if not path:
            return

        if major == 'IRP_MJ_DIRECTORY_CONTROL' and minor == 'IRP_MN_QUERY_DIRECTORY':
            self._discovered_paths.add(path)

        if major == 'IRP_MJ_WRITE':
            parent = _parent_path(path)
            if parent in self._discovered_paths:
                self._path_writes.setdefault(parent, set()).add(path)

    # ══════════════════════════════════════════════════════════════
    #  FEATURE 5 — Cross-class Writes
    # ══════════════════════════════════════════════════════════════

    def _cross_class_writes(self) -> float:
        """
        Restituisce 1.0 se il processo ha scritto su file con almeno 2 estensioni
        diverse, 0.0 altrimenti.  Un ransomware tipicamente cifra documenti di
        classi diverse (.docx, .pdf, .jpg, …).
        """
        return 1.0 if len(self._written_extensions) > 1 else 0.0

    def _update_extension_writes(self, r: dict):
        """Registra l'estensione del file di destinazione di ogni scrittura."""
        if r.get('major') == 'IRP_MJ_WRITE':
            path = r.get('dest_path') or ''
            ext = path.rsplit('.', 1)[-1].lower() if '.' in path else ''
            if ext:
                self._written_extensions.add(ext)

    # ══════════════════════════════════════════════════════════════
    #  FEATURE 6 — Write Frequency
    # ══════════════════════════════════════════════════════════════

    def _write_frequency(self, r: dict) -> float:
        """
        Proxy della frequenza di I/O: 1 / dt rispetto all'evento precedente.
        Alta frequenza indica attività intensa tipica di un'operazione di cifratura
        massiva. Il valore è normalizzato con sigmoid per restare in [0, 1].

        Restituisce 0.0 al primo evento di ogni PID.
        """
        t = r.get('time')
        if t is None or self._last_time is None:
            self._last_time = t
            return 0.0
        try:
            dt = (t - self._last_time).total_seconds()
        except (TypeError, AttributeError):
            dt = float(t) - float(self._last_time)
        self._last_time = t
        if dt <= 0:
            return 0.0
        # sigmoid(1/dt) normalizza il valore grezzo a (0, 1)
        raw = 1.0 / dt
        return float(torch.sigmoid(torch.tensor(raw)).item())


# ══════════════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════════════

def _parent_path(path: str) -> str:
    """Restituisce il parent directory di un path Windows (\\-separato)."""
    parts = path.split('\\')
    return '\\'.join(parts[:-1]) if len(parts) > 1 else path


# ══════════════════════════════════════════════════════════════════
#  TRAINING LOOP MINIMALE
# ══════════════════════════════════════════════════════════════════

def train(pid_rows: dict[int, list[dict]], epochs: int = 20, lr: float = 1e-3):
    """
    Allena il modello su un dizionario {pid: [rows]} dove ogni riga ha il campo
    'is_ransomware' (0/1).

    Args:
        pid_rows: dict pid → lista di dict (output di parser.load_irp_log + split_by_pid)
        epochs:   numero di epoche
        lr:       learning rate
    """
    model = Redemption()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        model.train()

        for pid, rows in pid_rows.items():
            if not rows:
                continue

            features = model.process_pid(rows)              # (N, 6)
            labels   = torch.tensor(
                [float(r.get('is_ransomware', 0)) for r in rows],
                dtype=torch.float32
            )                                               # (N,)

            optimizer.zero_grad()
            logits = model(features).squeeze(-1)            # (N,)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch:>3}/{epochs}  loss={total_loss / max(len(pid_rows), 1):.4f}")

    return model


if __name__ == '__main__':
    # Smoke-test: verifica che il forward pass funzioni senza errori
    model = Redemption()
    dummy = torch.rand(8, Redemption.N_FEATURES)   # batch di 8 campioni fittizi
    logits = model(dummy)
    assert logits.shape == (8, 1), f"Shape inattesa: {logits.shape}"
    print("Smoke-test superato:", logits.shape)
    print(model)