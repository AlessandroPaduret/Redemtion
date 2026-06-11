from __future__ import annotations

import io
from typing import Literal, Optional
import polars as pl
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════
#  MODELLO REDEMPTION
# ══════════════════════════════════════════════════════════════════


class Redemption(nn.Module):
    """
    Implementazione allenabile di Redemption (Kharraz et al., 2017) come nn.Module.
    Separa l'estrazione delle feature (gestita a monte tramite Polars) dalla
    combinazione lineare allenabile.
    """

    N_FEATURES = 6

    def __init__(self):
        super().__init__()
        # Classificatore lineare: combina le 6 feature in 1 logit
        self.fc = nn.Linear(self.N_FEATURES, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Applica il classificatore alle feature estratte.

        Args:
            features: Tensor di forma (N, 6)
        Returns:
            logit: Tensor di forma (N, 1) -> da usare con BCEWithLogitsLoss
        """
        return self.fc(features)


# ══════════════════════════════════════════════════════════════════
#  POLARS PIPELINE & DATALOADER
# ══════════════════════════════════════════════════════════════════


class PolarsRedemptionDataLoader:
    """
    DataLoader ottimizzato con Polars per caricare log IRP, ordinare per PID/Timestamp
    e generare finestre di campionamento (per operazioni o temporali).
    """

    FEATURE_COLS = [
        "EntropyRatio",
        "ContentOverwrite",
        "IsDelete",
        "DirTraversal",
        "TypeCoverage",
        "WriteFrequency",
    ]

    def __init__(self, source: str | io.StringIO):
        """
        Inizializza leggendo il file CSV tramite Polars.
        """
        # Carica il dataframe pulendo eventuali spazi bianchi nei nomi delle colonne o nei dati
        self.df = pl.read_csv(source).with_columns(
            [pl.col(pl.String).str.strip_chars()]
        )

    def create_windows(
        self,
        strategy: Literal["n-operations", "time-windows"],
        window_size: int = 5,
        time_window_secs: float = 10.0,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Suddivide il dataset in finestre pronte per l'addestramento o l'inferenza.

        Args:
            strategy: 'n-operations' o 'time-windows'
            window_size: numero di operazioni per finestra (usato se strategia è 'n-operations')
            time_window_secs: dimensione della finestra in secondi (usata se strategia è 'time-windows')
        """
        # Pre-processing iniziale comune
        working_df = self.df.with_columns(
            [
                # Converte IsDelete da booleano/stringa a float (0.0 o 1.0)
                pl.col("IsDelete").cast(pl.Float32),
                # Assicura il tipo numerico corretto per le feature e i target
                pl.col(
                    [
                        "EntropyRatio",
                        "ContentOverwrite",
                        "DirTraversal",
                        "TypeCoverage",
                        "WriteFrequency",
                    ]
                ).cast(pl.Float32),
                pl.col("GroundTruth").cast(pl.Float32),
                # Parsa il timestamp per l'ordinamento temporale corretto
                pl.col("Timestamp").str.to_time("%H:%M:%S%.f").alias("ParsedTime"),
            ]
        ).sort(["PID", "ParsedTime"])

        if strategy == "n-operations":
            # Crea un indice incrementale per ogni operazione all'interno dello stesso PID
            working_df = working_df.with_columns(
                pl.int_range(0, pl.len()).over("PID").alias("op_index")
            ).with_columns(
                # Divisione intera per assegnare l'ID della finestra
                (pl.col("op_index") // window_size).alias("window_id")
            )

        elif strategy == "time-windows":
            # Calcola il delta temporale in secondi rispetto al primo evento di ciascun PID
            working_df = working_df.with_columns(
                (
                    (pl.col("ParsedTime") - pl.col("ParsedTime").first())
                    .over("PID")
                    .dt.total_microseconds()
                    / 1e6
                ).alias("elapsed_seconds")
            ).with_columns(
                # Assegna l'ID della finestra basandosi sul tempo trascorso
                (pl.col("elapsed_seconds") // time_window_secs).alias("window_id")
            )
        else:
            raise ValueError(f"Strategia di windowing sconosciuta: {strategy}")

        # Raggruppa in modo aggregato mantenendo l'ordine cronologico delle sequenze
        grouped = working_df.group_by(["PID", "window_id"], maintain_order=True).agg(
            [pl.col(self.FEATURE_COLS), pl.col("GroundTruth")]
        )

        prepared_windows = []
        for row in grouped.iter_rows(named=True):
            # Estrae le liste aggregate dal record di Polars
            features_list = [row[f] for f in self.FEATURE_COLS]

            # Transpone la matrice da (6, N_operazioni) a (N_operazioni, 6)
            features_transposed = list(zip(*features_list))

            if not features_transposed:
                continue

            # Conversione diretta in Tensori PyTorch
            features_tensor = torch.tensor(features_transposed, dtype=torch.float32)
            labels_tensor = torch.tensor(row["GroundTruth"], dtype=torch.float32)

            prepared_windows.append((features_tensor, labels_tensor))

        return prepared_windows


# ══════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════


def train(
    windows: list[tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 20,
    lr: float = 1e-3,
) -> Redemption:
    """
    Allena il modello Redemption sulla lista di finestre generate dal loader.
    """
    model = Redemption()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        model.train()

        for features, labels in windows:
            if features.shape[0] == 0:
                continue

            optimizer.zero_grad()

            # Usiamo .view(-1) per gestire in modo robusto finestre di qualsiasi dimensione (anche N=1)
            logits = model(features).view(-1)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(windows), 1)
        print(f"Epoch {epoch:>2}/{epochs} | Loss media: {avg_loss:.4f}")

    return model


# ══════════════════════════════════════════════════════════════════
#  SMOKE TEST & VERIFICA FUNZIONALE
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Dati di esempio basati sul dump del log IRP fornito
    csv_data = """PID,Timestamp,Path,Type,EntropyRatio,ContentOverwrite,IsDelete,DirTraversal,TypeCoverage,WriteFrequency,GroundTruth
2136,10:50:23.915000000,c:\\users\\giuli\\documents\\documents\\services\\it\\cybersec,IRP_MJ_DIRECTORY_CONTROL,0.0,0.0,false,0.0,0.0,0.0,0
2136,10:50:24.115000000,c:\\users\\giuli\\documents\\documents\\services\\it\\cybersec,IRP_MJ_DIRECTORY_CONTROL,0.4,0.1,false,0.1,0.0,0.5,0
2136,10:50:24.515000000,c:\\users\\giuli\\documents\\documents\\services\\it\\servers,IRP_MJ_DIRECTORY_CONTROL,0.8,0.9,true,0.5,0.3,1.0,1
2136,10:50:24.915000000,c:\\users\\giuli\\documents\\documents\\services\\it\\servers,IRP_MJ_DIRECTORY_CONTROL,0.9,0.9,false,0.6,0.3,0.9,1
2136,10:50:35.015000000,c:\\users\\giuli\\documents\\documents\\services\\it\\servers,IRP_MJ_DIRECTORY_CONTROL,0.1,0.0,false,0.0,0.0,0.1,0
2136,10:50:35.215000000,c:\\users\\giuli\\documents\\documents\\services\\it\\servers\\102,IRP_MJ_DIRECTORY_CONTROL,0.2,0.0,false,0.1,0.0,0.2,0
"""

    print("--- Test di caricamento ed estrazione con Polars ---")
    # Inizializza il Loader (usando StringIO per simulare il file)
    loader = PolarsRedemptionDataLoader(io.StringIO(csv_data))

    # Opzione 1: Finestre basate sul numero di operazioni (es. blocchi di 3 operazioni)
    print("\n[Strategia: n-operations (Size=3)]")
    op_windows = loader.create_windows(strategy="n-operations", window_size=3)
    for i, (f, l) in enumerate(op_windows):
        print(
            f"Finestra {i}: Feature Shape {list(f.shape)} | Target Shape {list(l.shape)}"
        )

    # Opzione 2: Finestre basate sul tempo (es. intervalli di 5 secondi)
    print("\n[Strategia: time-windows (Duration=5.0s)]")
    time_windows = loader.create_windows(strategy="time-windows", time_window_secs=5.0)
    for i, (f, l) in enumerate(time_windows):
        print(
            f"Finestra {i}: Feature Shape {list(f.shape)} | Target Shape {list(l.shape)}"
        )

    print("\n--- Test di Addestramento ---")
    print("Avvio training su finestre temporali...")
    trained_model = train(time_windows, epochs=5, lr=1e-2)
    print("Modello addestrato con successo.")
