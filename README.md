
# Log parser per i log di cristiano 

rubato da cristiano:
- script.sh (per formattare il file out.log) per inserire tab dove non ci sono
- normalizer.py per ottenere un dataframe con le colonne corrette e i tipi di dato corretti ps. alla fine no

cose da chiedere:
- FileDispositionInformation esiste?? (dice se ho eliminato un file o no)
- le label is_ransomware??

TODO:
- implementare tutto con polars
- script bash che inizializza environment, installa dipendenze, scarica dataset(online), ecc

CONSIGLI:
- catturare scritture sequenziali
- catturare % di file sovrescritti
- extensioni dei file scritti è un po' scarna



Accuratezza: utile per classi bilanciate
$$(TP + TN) / (TP + TN + FP + FN)$$

Precisione: utile per minimizzare i falsi positivi
$$TP / (TP + FP)$$

Recall: utile per minimizzare i falsi negativi
$$ TP / (TP + FN)$$

F1-score: utile per bilanciare precisione e recall, soprattutto con classi sbilanciate
$$ 2 * (Precisione * Recall) / (Precisione + Recall)$$
