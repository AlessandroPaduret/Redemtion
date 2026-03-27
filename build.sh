# crea ambiente virtuale e installa dipendenze
python3 -m venv .venv
source .venv/bin/activate

# scarica il modulo per trovare dipendenze compatibili con la tua versione di Python

pip install --upgrade pip
pip install -r requirements.txt

# esegui training
python train.py