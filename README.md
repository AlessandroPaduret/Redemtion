
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
- implementare directory traversal non 1-1(1+n) ma file scritti / file della cartella presi da un file tipo quello in basso che si aggiorna( da mettere nel dataset principale assieme a IRP)
```
  files_per_directory:
    C:\Users\giuli\Documents: 1
    C:\Users\giuli\Documents\Companies: 1
    C:\Users\giuli\Documents\Companies\Amazon: 6
    C:\Users\giuli\Documents\Companies\CocaCola: 21
    C:\Users\giuli\Documents\Companies\Facebook: 19
```

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


# formato dei dati

nella cartella data ci sono tanti file tipo questo:
```csv
PID,Timestamp,Path,Type,EntropyRatio,ContentOverwrite,IsDelete,DirTraversal,TypeCoverage,WriteFrequency,GroundTruth
2136,10:50:23.915000000,c:\users\giuli\documents\documents\services\it\cybersec,IRP_MJ_DIRECTORY_CONTROL           ,0.0,0.0,false,0.0,0.0,0.0,0
2136,10:50:23.915000000,c:\users\giuli\documents\documents\services\it\cybersec,IRP_MJ_DIRECTORY_CONTROL           ,0.0,0.0,false,0.0,0.0,0.0,0
2136,10:50:23.915000000,c:\users\giuli\documents\documents\services\it\servers,IRP_MJ_DIRECTORY_CONTROL           ,0.0,0.0,false,0.0,0.0,0.0,0
2136,10:50:23.915000000,c:\users\giuli\documents\documents\services\it\servers,IRP_MJ_DIRECTORY_CONTROL           ,0.0,0.0,false,0.0,0.0,0.0,0
2136,10:50:23.915000000,c:\users\giuli\documents\documents\services\it\servers,IRP_MJ_DIRECTORY_CONTROL           ,0.0,0.0,false,0.0,0.0,0.0,0
2136,10:50:23.915000000,c:\users\giuli\documents\documents\services\it\servers\102,IRP_MJ_DIRECTORY_CONTROL           ,0.0,0.0,false,0.0,0.0,0.0,0
2136,10:50:23.915000000,c:\users\giuli\documents\documents\services\it\servers\102,IRP_MJ_DIRECTORY_CONTROL           ,0.0,0.0,false,0.0,0.0,0.0,0
2136,10:50:23.915000000,c:\users\giuli\documents\documents\services\it\servers\102,IRP_MJ_DIRECTORY_CONTROL           ,0.0,0.0,false,0.0,0.0,0.0,0
```
