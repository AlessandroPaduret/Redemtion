from __future__ import annotations

import io
import torch
import torch.nn as nn
from dataset import RedemptionDataLoader


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
    import io

    # Assicurati di importare PolarsLogProcessor se non è già importato in cima al file
    from dataset import PolarsLogProcessor

    # Dati di esempio basati sul dump del log IRP fornito
    csv_data = """PID,Timestamp,EntropyRatio,ContentOverwrite,IsDelete,DirTraversal,TypeCoverage,WriteFrequency,GroundTruth
2136,10:50:23.915000000,0.0,0.0,false,0.0,0.0,0.0,0
2136,10:50:24.115000000,0.4,0.1,false,0.1,0.0,0.5,0
2136,10:50:24.515000000,0.8,0.9,true,0.5,0.3,1.0,1
2136,10:50:24.915000000,0.9,0.9,false,0.6,0.3,0.9,1
2136,10:50:35.015000000,0.1,0.0,false,0.0,0.0,0.1,0
2136,10:50:35.215000000,0.2,0.0,false,0.1,0.0,0.2,0
"""

    print("══════════════════════════════════════════════════════════════════")
    print("  SMOKE TEST: ESTRAZIONE POLARS E MODELLO REDEMPTION")
    print("══════════════════════════════════════════════════════════════════\n")

    print("--- 1. Test di Estrazione con PolarsLogProcessor ---")

    # Inizializza il Processor (usando StringIO per simulare il file testuale)
    processor = PolarsLogProcessor(io.StringIO(csv_data))

    # Opzione 1: Finestre basate sul numero di operazioni
    print("\n[Strategia: n-operations (Size=3)]")
    op_windows = processor.create_windows(strategy="n-operations", window_size=3)
    for i, (features, labels) in enumerate(op_windows):
        print(
            f"  Finestra {i + 1}: Feature Shape {list(features.shape)} | Target Shape {list(labels.shape)}"
        )

    # Opzione 2: Finestre basate sul tempo
    print("\n[Strategia: time-windows (Duration=5.0s)]")
    time_windows = processor.create_windows(
        strategy="time-windows", time_window_secs=5.0
    )
    for i, (features, labels) in enumerate(time_windows):
        print(
            f"  Finestra {i + 1}: Feature Shape {list(features.shape)} | Target Shape {list(labels.shape)}"
        )

    print("\n--- 2. Test di Addestramento ---")
    print("Avvio training sulle finestre temporali (time-windows)...")
    trained_model = train(time_windows, epochs=5, lr=1e-2)
    print("Modello addestrato con successo.")

    print("\n--- 3. Test di Inferenza (Forward Pass) ---")
    trained_model.eval()
    with torch.no_grad():
        # Prendiamo le feature della prima finestra creata
        sample_features, sample_labels = time_windows[0]

        # Facciamo passare i dati nel modello
        sample_logits = trained_model(sample_features)

        # Applichiamo la sigmoide per ottenere probabilità tra 0 e 1
        sample_probs = torch.sigmoid(sample_logits).squeeze(-1)

        print(f"Predizioni (Probabilità) per gli eventi della Finestra 1:")
        for pred, true in zip(sample_probs, sample_labels):
            print(
                f"  Label Reale: {true.item():.0f} -> Predizione Modello: {pred.item():.4f}"
            )

    print("\nScript funzionante. Pipeline verificata.")
