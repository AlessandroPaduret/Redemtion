
# Log parser per i log di cristiano 

rubato da cristiano:
- script.sh (per formattare il file out.log) per inserire tab dove non ci sono
- normalizer.py per ottenere un dataframe con le colonne corrette e i tipi di dato corretti ps. alla fine no

cose da chiedere:
- FileDispositionInformation esiste?? (dice se ho eliminato un file o no)
- le label is_ransomware??

TODO:
- implementare tutto con polars
- scrivere codice più carino e più efficiente possibile

CONSIGLI:
- catturare scritture sequenziali
- catturare % di file sovrescritti
- extensioni dei file scritti è un po' scarna