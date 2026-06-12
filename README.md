# Redemtion

Linear model per ransomware detection su operazioni IRP.

## Panoramica
Questo progetto contiene il codice per allenare un modello lineare (PyTorch / PyTorch-like) per rilevare attività ransomware su operazioni IRP. Il flusso di lavoro è intenzionalmente semplice:

- Preparare il dataset (vedi sotto).
- Mettere tutti i file del dataset in una cartella (es. `data/`).
- Lanciare l'allenamento con il comando `uv run train.py`.
- Il modello migliore verrà salvato in `redemption_best.pt`.

> Nota: per ottenere il dataset di esempio si usa la repository: https://github.com/N-T-Lock/Progetto-Cybersecurity/  
Basta scaricare/clone la repo e copiare i file dati nella cartella del progetto (es. `data/`).

## Requisiti
- Python 3.8+ (o compatibile).
- Le dipendenze sono dichiarate in `pyproject.toml`.
- Assicurati di avere la CLI `uv` disponibile perché il comando di esecuzione usa `uv run train.py`. 
- Consiglio: `uv` crea l'ambiente virtuale in automatico quindi basterebbe preoccuparsi di installare lui e usarlo. Le altre dipendeze può risolverle lui.

## Struttura dei dati
I file di input devono essere CSV con le seguenti colonne (header):

PID,Timestamp,EntropyRatio,ContentOverwrite,IsDelete,DirTraversal,TypeCoverage,WriteFrequency,GroundTruth

Descrizione colonne:
- PID: identificatore del processo (intero)
- Timestamp: orario dell'evento (es. `10:50:23.915000000`)
- EntropyRatio: valore numerico (float) che misura il cambiamento di entropia
- ContentOverwrite: valore numerico (float) relativo a sovrascritture di contenuto
- IsDelete: booleano (`true`/`false`) che indica se c'è stata un'operazione di delete
- DirTraversal: valore numerico (float) indicatore di traversal directory
- TypeCoverage: valore numerico (float) che misura copertura tipologie di file
- WriteFrequency: valore numerico (float) che misura la frequenza di scrittura
- GroundTruth: etichetta (0 = benigno, 1 = ransomware) — usata per l'allenamento supervisionato

Esempio (righe di esempio):
```
2136,10:50:23.915000000,0.0,0.0,false,0.0,0.0,0.0,0
2136,10:50:23.915000000,0.0,0.0,false,0.0,0.0,0.0,0
2136,10:50:23.915000000,0.0,0.0,false,0.0,0.0,0.0,0
2136,10:50:23.915000000,0.0,0.0,false,0.0,0.0,0.0,0
```

Assicurati che i CSV abbiano l'header e che la colonna `GroundTruth` sia presente per l'allenamento.

## Dove mettere i dati
Copia i file CSV estratti da https://github.com/N-T-Lock/Progetto-Cybersecurity/ nella cartella `data/` (o in un'altra cartella a tua scelta). Se il progetto richiede uno specifico nome di file o percorso, adattalo di conseguenza. L'importante è che lo script `train.py` riesca a trovare e leggere i CSV.

Esempio:
```
project-root/
├─ train.py
├─ data/
│  ├─ dataset1.csv
│  └─ dataset2.csv
└─ pyproject.toml
```

## Come allenare (semplice)
1. Metti molti dati nel folder `data/` (più esempi, meglio il modello).
2. Installa le dipendenze come descritto nella sezione Requisiti.
3. Esegui:
```bash
uv run train.py
```
Questo avvierà il processo di training. Lo script `train.py` dovrebbe occuparsi di:
- caricare i CSV,
- preprocessare (es. gestione tipi, normalizzazione),
- dividere train/validation,
- allenare il modello,
- salvare il modello migliore su disco.

Al termine, il modello migliore sarà salvato in un file chiamato `file.pt` (o percorso equivalente definito nello script).

## Suggerimenti rapidi
- Bilancia i dati se la classe `GroundTruth` è sbilanciata (over/undersampling o pesi di classe).
- Normalizza o scala le feature numeriche (es. StandardScaler).
- Controlla che i valori booleani (`IsDelete`) siano convertiti correttamente (0/1) prima di passare al modello.
- Valuta il modello su un set di test separato e salva le metriche principali (accuracy, precision, recall, F1).

## Contatti / note
Per qualsiasi dettaglio sul dataset fai riferimento alla repo di origine: https://github.com/N-T-Lock/Progetto-Cybersecurity/  
