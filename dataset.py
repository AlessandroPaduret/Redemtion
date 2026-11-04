import os
import io
import glob
import random
import tempfile
import torch
from typing import Literal
import polars as pl
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ══════════════════════════════════════════════════════════════════
#  1. DATA PROCESSOR (Polars)
# ══════════════════════════════════════════════════════════════════


class PolarsLogProcessor:
    """
    Processore ottimizzato con Polars per estrarre le feature e dividere
    i log IRP in finestre temporali o basate su operazioni.
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
        self.df = pl.read_csv(source).with_columns(
            [pl.col(pl.String).str.strip_chars()]
        )

    def create_windows(
        self,
        strategy: Literal["n-operations", "time-windows", "state-windows"],
        window_size: int = 5,
        time_window_secs: float = 10.0,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:

        working_df = self.df.with_columns(
            [
                pl.col("IsDelete").cast(pl.Float32),
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
                pl.col("Timestamp").str.to_time("%H:%M:%S%.f").alias("ParsedTime"),
            ]
        ).sort(["PID", "ParsedTime"], maintain_order=True)

        if strategy == "state-windows":
            # Finestre già calcolate con uno stato separato ciascuna (strategia
            # Windowed di Progetto-Cybersecurity): una finestra = un campione.
            if "WindowStart" not in working_df.columns:
                raise ValueError(
                    "state-windows richiede la colonna WindowStart "
                    "(CSV estratti con 'Redemption a finestre di stato')"
                )
            working_df = working_df.with_columns(pl.col("WindowStart").alias("window_id"))
        elif strategy == "n-operations":
            working_df = working_df.with_columns(
                pl.int_range(0, pl.len()).over("PID").alias("op_index")
            ).with_columns((pl.col("op_index") // window_size).alias("window_id"))
        elif strategy == "time-windows":
            working_df = working_df.with_columns(
                (
                    (pl.col("ParsedTime") - pl.col("ParsedTime").first())
                    .over("PID")
                    .dt.total_microseconds()
                    / 1e6
                ).alias("elapsed_seconds")
            ).with_columns(
                (pl.col("elapsed_seconds") // time_window_secs).alias("window_id")
            )
        else:
            raise ValueError(f"Strategia sconosciuta: {strategy}")

        grouped = working_df.group_by(["PID", "window_id"], maintain_order=True).agg(
            [pl.col(self.FEATURE_COLS), pl.col("GroundTruth")]
        )

        prepared_windows = []
        for row in grouped.iter_rows(named=True):
            features_list = [row[f] for f in self.FEATURE_COLS]
            features_transposed = list(zip(*features_list))

            if not features_transposed:
                continue

            features_tensor = torch.tensor(features_transposed, dtype=torch.float32)
            labels_tensor = torch.tensor(row["GroundTruth"], dtype=torch.float32)
            prepared_windows.append((features_tensor, labels_tensor))

        return prepared_windows


# ══════════════════════════════════════════════════════════════════
#  2. PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════


class RedemptionDataset(Dataset):
    """
    Dataset PyTorch che aggrega le finestre generate da multipli file CSV.
    """

    def __init__(
        self,
        file_paths: list[str],
        strategy: str,
        window_size: int = 5,
        time_window_secs: float = 10.0,
    ):
        self.windows = []
        for path in file_paths:
            processor = PolarsLogProcessor(path)
            file_windows = processor.create_windows(
                strategy=strategy,
                window_size=window_size,
                time_window_secs=time_window_secs,
            )
            self.windows.extend(file_windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


# ══════════════════════════════════════════════════════════════════
#  3. PYTORCH DATALOADER CUSTOM (con padding zeri)
# ══════════════════════════════════════════════════════════════════


def pad_collate_fn(batch):
    """
    Prende le finestre di dimensione variabile dal batch e applica
    il padding con zeri per allinearle tutte alla finestra più lunga.
    """
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # padding_value=0.0 riempie le differenze di lunghezza con zeri
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0.0)

    return features_padded, labels_padded


class RedemptionDataLoader(DataLoader):
    """
    DataLoader custom che eredita dallo standard PyTorch.
    Gestisce automaticamente il padding delle sequenze.
    """

    def __init__(
        self,
        dataset: RedemptionDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=pad_collate_fn,  # <--- Qui avviene la magia dei padding
            **kwargs,
        )

    @classmethod
    def create_splits_from_folder(
        cls,
        data_dir: str,
        val_ratio: float = 0.2,
        batch_size: int = 32,
        seed: int = 42,
        strategy: str = "time-windows",
        window_size: int = 5,
        time_window_secs: float = 10.0,
    ) -> tuple["RedemptionDataLoader", "RedemptionDataLoader"]:

        all_files = glob.glob(os.path.join(data_dir, "*.csv"))
        all_files.sort()

        if not all_files:
            raise ValueError(f"Nessun file .csv trovato in {data_dir}")

        random.seed(seed)
        random.shuffle(all_files)

        split_idx = int(len(all_files) * (1 - val_ratio))
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]

        train_dataset = RedemptionDataset(
            train_files, strategy, window_size, time_window_secs
        )
        val_dataset = RedemptionDataset(
            val_files, strategy, window_size, time_window_secs
        )

        train_loader = cls(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = cls(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════
#  4. MAIN BLOCK PER TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("--- Avvio Test del DataLoader Custom ---\n")

    # Creiamo dati dummy per testare la pipeline
    csv_content_1 = """PID,Timestamp,Path,Type,EntropyRatio,ContentOverwrite,IsDelete,DirTraversal,TypeCoverage,WriteFrequency,GroundTruth
100,10:00:01.000,C:\\test,IRP,0.1,0.1,false,0.0,0.1,0.1,0
100,10:00:03.000,C:\\test,IRP,0.2,0.2,false,0.0,0.1,0.2,0
100,10:00:15.000,C:\\test,IRP,0.9,0.9,true,0.8,0.9,0.9,1
"""
    # Questo file avrà più eventi vicini per generare una finestra più grande
    csv_content_2 = """PID,Timestamp,Path,Type,EntropyRatio,ContentOverwrite,IsDelete,DirTraversal,TypeCoverage,WriteFrequency,GroundTruth
200,10:00:01.000,C:\\test,IRP,0.1,0.1,false,0.0,0.1,0.1,0
200,10:00:02.000,C:\\test,IRP,0.2,0.2,false,0.0,0.1,0.2,0
200,10:00:03.000,C:\\test,IRP,0.3,0.3,false,0.0,0.1,0.3,0
200,10:00:04.000,C:\\test,IRP,0.4,0.4,false,0.0,0.1,0.4,0
"""

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Salviamo i file finti nella cartella temporanea
        with open(os.path.join(tmpdirname, "file1.csv"), "w") as f:
            f.write(csv_content_1)
        with open(os.path.join(tmpdirname, "file2.csv"), "w") as f:
            f.write(csv_content_2)

        print(f"Cartella temporanea creata con 2 file: {tmpdirname}")

        # Inizializziamo i Dataloader (usiamo val_ratio=0.5 per forzarne uno in train e uno in val)
        train_loader, val_loader = RedemptionDataLoader.create_splits_from_folder(
            data_dir=tmpdirname,
            val_ratio=0.5,
            batch_size=2,  # Prende fino a 2 finestre alla volta
            strategy="time-windows",
            time_window_secs=10.0,
        )

        print(
            f"Dataset caricato: {len(train_loader.dataset)} finestre in Train, {len(val_loader.dataset)} in Val.\n"
        )

        print("--- Ispezione Batch e Zeri (Padding) ---")
        for batch_idx, (features, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx + 1}")
            print(
                f"Forma Features: {list(features.shape)} -> (Batch, Length, Features)"
            )
            print(f"Forma Labels:   {list(labels.shape)} -> (Batch, Length)")
            print(
                "Primo elemento delle features nel batch (nota gli zeri finali se c'è padding):"
            )
            print(features[0])
            break
