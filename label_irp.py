
import pandas as pd

read_csv = pd.read_csv('data/IRP/parsed_irp_log.csv')

# devo affiungere la colonna is_ransomware se il pid è questo 00160 allora 1 altrimenti 0

label = pd.Series([1 if pid == 160 else 0 for pid in read_csv['pid']])

# aggiungo la colonna al dataframe
read_csv['is_ransomware'] = label

# salvo il nuovo dataframe in un nuovo csv
read_csv.to_csv('data/IRP/labeled_irp.csv', index=False)