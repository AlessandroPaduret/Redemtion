"""
irp_parser.py
─────────────
Parser robusto per log IRP/FIO basato su regex.

Colonne finali:
  type | time | pid | major | minor | source_path
  presize | postsize | offset | fileinfoclass
  filenamelength | newfilename | buffersize
  entropy | dest_path
"""

import re
import pandas as pd
from typing import Optional

# ══════════════════════════════════════════════════════════════════
#  PATTERN REGEX
# ══════════════════════════════════════════════════════════════════

RE_TYPE    = re.compile(r'^(IRP|FIO)')
RE_MAJOR   = re.compile(r'(IRP_MJ_\w+)')
RE_MINOR   = re.compile(r'(IRP_MN_\w+)')

# \Device\HarddiskVolume...\process.exe
RE_SRC = re.compile(r'(\\Device\\[^\t]+)')

# Campi NOMINATI ─ robusti a spaziatura variabile
RE_PRESIZE   = re.compile(r'PreSize\s+(\d+)',   re.IGNORECASE)
RE_POSTSIZE  = re.compile(r'PostSize\s+(\d+)',  re.IGNORECASE)
RE_OFFSET    = re.compile(r'\bOffset\s+(\d+)',  re.IGNORECASE)

# FileInfoClass: qualsiasi token CamelCase che termina in "Information" File*Information
RE_FILEINFOCLASS = re.compile(r'\b(File\w+Information)\b')

# FilenameLength — gestisce il typo "FilenameLangth" (e simili varianti)
RE_FNAMELEN  = re.compile(r'Filename\w{0,3}[Ll]e?[an]ng?th\s+(\d+)', re.IGNORECASE)

# NewFileName – cattura tutto fino al prossimo tab o fine stringa
RE_NEWFILENAME = re.compile(r'NewFileName\s+(\S+)', re.IGNORECASE)

# Entropy: float con ≥6 decimali  (es. 7.225866931622)
RE_ENTROPY   = re.compile(r'\b(\d+\.\d{6,})\b')

# Dest path: C:\…  oppure  \??\c:\…  — ultimo della riga
RE_DESTPATH  = re.compile(r'([A-Za-z]:\\[^\t\n]*|\\?\?\\[^\t\n]+)')

# Intero nudo rimasto dopo aver consumato tutti i campi nominati
# (buffersize / status code)
RE_BARE_INT  = re.compile(r'(?<!\d)\b(\d+)\b(?!\d)')


# ══════════════════════════════════════════════════════════════════
#  PARSER SINGOLA RIGA
# ══════════════════════════════════════════════════════════════════

def parse_line(line: str) -> Optional[dict]:
    """
    Parsa una singola riga del log e restituisce un dict con tutte le colonne.
    Restituisce None se la riga non è un record IRP/FIO valido.
    Input: riga di testo (con eventuale newline finale)
    Output: dict con chiavi:
  type | time | pid | major | minor | source_path
  presize | postsize | offset | fileinfoclass
  filenamelength | newfilename | buffersize
  entropy | dest_path
    """

    line = line.strip()
    if not RE_TYPE.match(line):   # ← questo non va tolto
        return None
    
    parts = [p.strip() for p in line.split('\t') if p.strip()]

    rec: dict = {k: None for k in [
        'type', 'time', 'pid', 'major', 'minor', 'source_path',
        'presize', 'postsize', 'offset', 'fileinfoclass',
        'filenamelength', 'newfilename',
        'buffersize', 'entropy', 'dest_path'
    ]}

    # ── 1. Campi fissi di testa ──────────────────────────────────

    # il prefisso sicuro è sempre: type, time, pid, major
    rec['type'] = RE_TYPE.search(parts[0]).group(1) if RE_TYPE.search(parts[0]) else None
    rec['time'] = parts[1]
    rec['pid']  = parts[2]
    rec['major'] = parts[3]

    # il suffisso sicuro è sempre: buffersize, entropy, dest_path (ultimo percorso Windows)
    rec['dest_path'] = parts[-1]
    rec['entropy'] = float(parts[-2])
    rec['buffersize'] = int(parts[-3])

    idx = 4  # indice per campi variabili dopo major

    # minor (opzionale)
    if RE_MINOR.match(parts[idx]):
        rec['minor'] = parts[idx]; idx += 1

    # source_path obbligatorio (sempre presente)
    rec['source_path'] = parts[idx]


    # ── 2. Coda: tutto il resto come stringa unica ───────────────
    #    Usiamo i tab come separatori ma vogliamo applicare regex
    #    su un testo "piatto" per comodità.
    rest = '\t'.join(parts[idx+1:-3])  # tutto dopo major fino a buffersize escluso

    rec['presize'] = int(RE_PRESIZE.search(rest).group(1)) if RE_PRESIZE.search(rest) else None
    rec['postsize'] = int(RE_POSTSIZE.search(rest).group(1)) if RE_POSTSIZE.search(rest) else None
    rec['offset'] = int(RE_OFFSET.search(rest).group(1)) if RE_OFFSET.search(rest) else None
    rec['fileinfoclass'] = RE_FILEINFOCLASS.search(rest).group(1) if RE_FILEINFOCLASS.search(rest) else None
    rec['filenamelength'] = int(RE_FNAMELEN.search(rest).group(1)) if RE_FNAMELEN.search(rest) else None
    rec['newfilename'] = RE_NEWFILENAME.search(rest).group(1) if RE_NEWFILENAME.search(rest) else None    

    return rec


# ══════════════════════════════════════════════════════════════════
#  CARICAMENTO FILE INTERO → DataFrame
# ══════════════════════════════════════════════════════════════════

def load_irp_log(filepath: str) -> pd.DataFrame:
    records = []
    skipped = 0
    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        # apparte la prima riga parsa tutto
        next(fh)  # salta la prima riga
        for line in fh:
            r = parse_line(line)
            if r:
                records.append(r)
            elif line.strip():
                skipped += 1

    print(f"[load_irp_log] parsed={len(records)}  skipped={skipped}")
    df = pd.DataFrame(records)
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S:%f', errors='coerce')
    df['pid']  = df['pid'].astype(str)
    return df

def optimize_dtypes(df):
    # Esempio di ottimizzazione
    if 'type' in df.columns:
        df['type'] = df['type'].astype('category')
    if 'pid' in df.columns:
        df['pid'] = pd.to_numeric(df['pid'], downcast='unsigned')
    # Entropy è un float, ma float32 basta e avanza
    if 'entropy' in df.columns:
        df['entropy'] = df['entropy'].astype('float64')
    return df

def load_irp_mem_safe(filepath, chunk_lines=100000):
    reader = pd.read_csv(
    filepath, 
    sep='|', # Un separatore che non usi (o metti qualcosa di improbabile)
    names=['raw'], 
    header=None, 
    skiprows=1, 
    quoting=3, 
    lineterminator='\n', 
    chunksize=chunk_lines
    )
    
    final_chunks = []
    
    for i, chunk in enumerate(reader):
        print(f"Processando blocco {i}...")
        # Applichi il parsing solo su questo pezzetto
        parsed_chunk = chunk['raw'].apply(parse_line).dropna()
        
        # Trasformi subito in DataFrame e ottimizzi i tipi
        df_chunk = pd.DataFrame(list(parsed_chunk))
        df_chunk = optimize_dtypes(df_chunk) # Funzione sotto
        
        final_chunks.append(df_chunk)
    
    # Unisci tutto alla fine (o scrivi su un nuovo file se è troppo grande)
    return pd.concat(final_chunks, ignore_index=True)


def add_dt(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge dt (secondi dall'evento precedente) per ogni PID."""
    df = df.copy().sort_values(['pid', 'time'])
    df['dt'] = df.groupby('pid')['time'].diff().dt.total_seconds().fillna(0.0)
    return df


def split_by_pid(df: pd.DataFrame) -> dict:
    return {pid: grp.reset_index(drop=True) for pid, grp in df.groupby('pid')}


# ══════════════════════════════════════════════════════════════════
#  UNIT TEST
# ══════════════════════════════════════════════════════════════════

EXAMPLES = [
    # 1) SET_INFORMATION + FileBasicInformation
    (
        "IRP\t17:59:47:536\t00160\tIRP_MJ_SET_INFORMATION             \t"
        r"\Device\HarddiskVolume2\Users\giuli\Desktop\cerber_samples\00c73cd.exe"
        "\tFileBasicInformation\t0\t0.000000000000\t"
        r"C:\Users\giuli\Documents\Companies\Amazon\des.jpg",
        {
            'type': 'IRP', 'minor': None,
            'major': 'IRP_MJ_SET_INFORMATION',
            'fileinfoclass': 'FileBasicInformation',
            'presize': None, 'postsize': None, 'offset': None,
            'filenamelength': None, 'newfilename': None,
            'buffersize': 0, 'entropy': 0.0,
            'dest_path': r'C:\Users\giuli\Documents\Companies\Amazon\des.jpg',
        },
    ),
    # 2) SET_INFORMATION + FileRenameInformation (FilenameLength + NewFileName)
    (
        "IRP\t17:59:47:520\t00160\tIRP_MJ_SET_INFORMATION             \t"
        r"\Device\HarddiskVolume2\Users\giuli\Desktop\cerber_samples\00c73cd.exe"
        "\tFileRenameInformation\tFilenameLangth 61\t"
        r"NewFileName \??\c:\users\giuli\documents\companies\amazon\lZ0tBU9lx8.834f"
        "\t0\t0.000000000000\t"
        r"C:\Users\giuli\Documents\Companies\Amazon\0001.png",
        {
            'type': 'IRP', 'minor': None,
            'major': 'IRP_MJ_SET_INFORMATION',
            'fileinfoclass': 'FileRenameInformation',
            'filenamelength': 61,
            'newfilename': r'\??\c:\users\giuli\documents\companies\amazon\lZ0tBU9lx8.834f',
            'presize': None, 'postsize': None, 'offset': None,
            'buffersize': 0, 'entropy': 0.0,
            'dest_path': r'C:\Users\giuli\Documents\Companies\Amazon\0001.png',
        },
    ),
    # 3) FIO + IRP_MJ_WRITE + IRP_MN_NORMAL  (PreSize, PostSize, Offset, buffersize)
    (
        "FIO\t17:59:47:520\t00160\tIRP_MJ_WRITE                       \t"
        "IRP_MN_NORMAL                      \t"
        r"\Device\HarddiskVolume2\Users\giuli\Desktop\cerber_samples\00c73cd.exe"
        "\tPreSize 48712\tPostSize 48968\tOffset 48712\t256\t7.225866931622\t"
        r"C:\Users\giuli\Documents\Companies\Amazon\0001.png",
        {
            'type': 'FIO', 'minor': 'IRP_MN_NORMAL',
            'major': 'IRP_MJ_WRITE',
            'presize': 48712, 'postsize': 48968, 'offset': 48712,
            'fileinfoclass': None, 'filenamelength': None, 'newfilename': None,
            'buffersize': 256, 'entropy': 7.225866931622,
            'dest_path': r'C:\Users\giuli\Documents\Companies\Amazon\0001.png',
        },
    ),
    (
        "IRP\t17:59:11:027\t03980\tIRP_MJ_READ                         \t"
        "IRP_MN_NORMAL                       \t"
        r"\Device\HarddiskVolume2\Windows\explorer.exe"
        "\tPreSize 402\tPostSize 402\tOffset 0\t404\t3.480763049601\t"
        r"C:\Users\giuli\Documents\desktop.ini",
        {
            'type': 'IRP', 'minor': 'IRP_MN_NORMAL',
            'major': 'IRP_MJ_READ',
            'presize': 402, 'postsize': 402, 'offset': 0,
            'fileinfoclass': None, 'filenamelength': None, 'newfilename': None,
            'buffersize': 404, 'entropy': 3.480763049601,
            'dest_path': r'C:\Users\giuli\Documents\desktop.ini',
        }
    ),
    (
        "IRP\t17:59:39:833\t03980\tIRP_MJ_DIRECTORY_CONTROL            \tIRP_MN_QUERY_DIRECTORY              \t"
        r"\Device\HarddiskVolume2\Windows\explorer.exe"
        " \t FileIdBothDirectoryInformation\t0\t0.000000000000\t"
        r"C:\Users\giuli\Documents"
        , {
            'type': 'IRP', 'minor': 'IRP_MN_QUERY_DIRECTORY',
            'major': 'IRP_MJ_DIRECTORY_CONTROL',
            'presize': None, 'postsize': None, 'offset': None,
            'fileinfoclass': 'FileIdBothDirectoryInformation',
            'filenamelength': None, 'newfilename': None,
            'buffersize': 0, 'entropy': 0.0,
            'dest_path': r'C:\Users\giuli\Documents',
        }
    )
]


def run_tests():
    print("══ UNIT TEST ══════════════════════════════════════════════")
    all_ok = True
    for i, (line, expected) in enumerate(EXAMPLES):
        rec = parse_line(line)
        if rec is None:
            print(f"  [FAIL] esempio {i}: parse_line → None")
            all_ok = False
            continue

        fails = []
        for k, v in expected.items():
            got = rec.get(k)
            if isinstance(v, float) and v != 0.0:
                ok = got is not None and abs(got - v) < 1e-9
            else:
                ok = (got == v)
            if not ok:
                fails.append(f"    {k}: atteso={v!r}  ottenuto={got!r}")

        if fails:
            print(f"  [FAIL] esempio {i} ({expected['major']}):")
            for f in fails:
                print(f)
            all_ok = False
        else:
            print(f"  [OK]   esempio {i} — {expected['major']}")

    print()
    print("══ RISULTATO:", "TUTTI OK ✓" if all_ok else "CI SONO ERRORI ✗")
    print()

    # Stampa il record completo dell'esempio più ricco
    print("── Record completo esempio 3 (WRITE) ─────────────────────")
    rec = parse_line(EXAMPLES[2][0])
    for k, v in rec.items():
        print(f"  {k:<18} = {v!r}")


if __name__ == "__main__":
    run_tests()

    df = load_irp_log("data/IRP/good_out.log")
    print(df.head())

    df.to_csv("data/IRP/parsed_irp_log.csv", index=False)

    print(df.iloc[1].to_dict())

    pass

    # Utilizzo reale:
    # df = load_irp_log("data/IRP/out.log")
    # df = add_dt(df)
    # pid_dfs = split_by_pid(df)