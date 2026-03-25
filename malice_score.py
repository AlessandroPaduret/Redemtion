
def malice_score(weighs: dict, r: dict) -> int:

    score = 0
    for i in range(1,6):
        score += weighs[i] * r[i]
    
    return score

# Gli oggetti di tipo redempion possono calcolare il malice score
# Questi oggetti "mangiano" righe di log restituire da parser.py calcola le feature e restituiscono un malice score
# il problema è che le feature non sono stateless ma dipendono dalle operazioni precedenti

class Redemption:
    
    def __init__(self, weighs: dict):
        self.weighs = weighs
        self.read_operations = {} # chiave: file_name + file_offset, valore: entropia dell'ultima operazione di lettura su quel file offset
        self.file_writes = {} # chiave: file_name, valore: set di buffer unici scritti su quel file (identificati dall'offset)
        self.file_dimensions = {} # chiave: file_name, valore: dimensione del file iniziale

    
    def score(self, r: dict) -> int:
        return malice_score(self.weighs, r)
    
    def get(self, r: dict) -> int:
        """
        il malice score viene calcolato a partire dall'input e da alcune feature che dipendono dalle operazioni precedenti
        quindi trattiamo l'input come stream e aggiorniamo le feature che dipendono dalle operazioni precedenti ogni volta che leggiamo una nuova riga di log
        Input: riga di log (r) con i campi necessari per updatare le feature e calcolare il malice score
        Output: malice score calcolato a partire dall'input e dalle feature aggiornate
        """
        feature = []
        # calcolo entropy ratio e aggiorno le operazioni di lettura
        self.update_read_operations(r)
        ratio = self.entropy_ratio(r)
        feature.append(ratio)

        # calcolo file overwrite e aggiorno le operazioni di scrittura
        self.update_file_overwrite(r)
        overwrite = self.file_overwrite(r)
        feature.append(overwrite)

        # calcolo delete operations 1 se delete è true, 0 altrimenti
        feature.append(1 if r['major'] == 'IRP_MJ_SET_INFORMATION' and r['fileinfoclass'] == 'FileDispositionInformation' and r['delete'] == True else 0)

        # calcolo file sequential writes e aggiorno le operazioni di scrittura
        self.update_file_sequential_writes(r)
        sequential_writes = self.file_sequential_writes(r)
        feature.append(sequential_writes)

        # calcolo di quante estensioni di file diversi ha scritto un certo pid(assumo che loggo il log di un pid alla volta)
        self.update_extension_writes(r)
        extension_writes = self.extension_writes(r)
        feature.append(extension_writes)

        # calcoliamo r6 formula: r6 = 1/dt dove dt è il tempo trascorso tra l'ultima operazione di scrittura su un file e l'operazione di scrittura corrente su quel file, se il file è lo stesso
        r6 = self.dt(r)
        feature.append(r6)

        return self.score(feature)

    def entropy_ratio(self, current_op: dict) -> float:
        """
        Entropy Ratio of Data Blocks. For every read and write request to a file,
        Redemption computes the entropy of the corresponding data buffers in
        the I/O traces similar to. Comparing the entropy of read and write requests
        to and from the same file offset serves as an excellent indicator of ransomware
        behavior. This is due to the popular strategy of reading in the original file data,
        encrypting it, and writing the encrypted version.
        
        The process that changes the entropy of the data blocks between a read
        and a write request to a higher value receives a higher malice score. The required
        value is calculated as an additive inverse of the entropy value of read and write
        ratio, and resides on [0,1], meaning that the higher the value of entropy in the
        write operation, the closer the value of the entropy to 1. If the entropy of the
        data block in write is smaller than the read operation, we assign the value 0 to
        this feature.
        """
        if len(self.read_operations) == 0:
            return 0
        
        # calcoliamo l'entropy ratio come il rapporto tra l'entropia dell ultima operazione di lettura / current_op['entropy'] per un determitato file offset e file name
        '''
        for op in self.read_operations:
            if op['file_offset'] == current_op['file_offset'] and op['file_name'] == current_op['file_name']:
                return self.last_read_entropy / current_op['entropy']
        return 0
        '''
        # okk ma vorrei farlo efficiente
        # possiamo tenere un dizionario con chiave file_name + file_offset e valore l'entropia dell'ultima operazione di lettura su quel file offset
        # in questo modo possiamo aggiornare l'entropia dell'ultima operazione di lettura ogni volta che leggiamo una nuova riga di log e calcolare l'entropy ratio in
        # tempo costante
        key = current_op['dest_path'] + str(current_op['offset'])
        if key in self.read_operations and self.read_operations[key] > current_op['entropy']:
            return self.read_operations[key] / current_op['entropy']
        return 0
    
    def update_read_operations(self, r: dict):
        # aggiorniamo le operazioni di lettura se l'operazione corrente è una lettura
        if r['major'] == 'IRP_MJ_READ':
            key = r['dest_path'] + str(r['offset'])
            self.read_operations[key] = r.get('entropy', 0)

    def file_overwrite(self, r: dict):
        """
        File Content Overwrite. Redemption monitors how a process requests write
        access to data blocks. In a typical ransomware attack, in order to minimize the
        chance of recovering files, the malicious process overwrites the content of the
        user files with random data. Our system increases the malice score of a process
        as the process requests write access to different parts of a file. In fact, a process
        is assigned a higher malice score if it overwrites all the content of the files.

        (r2): If a process iterates over the content of a file with write privilege, it will
        receive a higher malice score. If the size of the file A is sA, and yA is the total
        size of the data blocks modified by the process, the feature is calculated as yA / sA
        where the higher the number of data blocks modified by the process, the closer
        the value is to 1.
        """
        # per calcolare questa feature dobbiamo tenere traccia delle operazioni di scrittura sui file
        # 
        # sum (buffer unici scritti su un file) / size del file
        # possiamo tenere un dizionario con chiave file_name e valore un set di buffer unici scritti su quel file
        # la size del file iniziale la salviamo quando incontriamo la prima operazione di scrittura su quel file

        sum_buffs = 0
        for b in self.file_dimensions[r['dest_path']]:
            sum_buffs += b["size"]
        
        return sum_buffs / self.file_dimensions[r['dest_path']]

    def update_file_overwrite(self, r: dict):
        """
        aggiorniamo le operazioni di scrittura se l'operazione corrente è una scrittura, salviamo l'offset e la dimensione del buffer scritto su quel file
        se il file non è ancora stato incontrato, salviamo anche la dimensione del file iniziale per calcolare la feature file overwrite
        """
        if r['major'] == 'IRP_MJ_WRITE':
            if r['dest_path'] not in self.file_dimensions:
                self.file_writes[r['dest_path']] = set()
                self.file_dimensions[r['dest_path']] = r['presize']
            # salviamo con identificatore l'offset e basta
            self.file_dimensions[r['dest_path']].add({'offset': r['offset'], 'size': r['buffer_size']})


    def update_file_dim_rename(self, r: dict):
        """
        File Rename. Redemption monitors the file rename operations performed by a process. In a
        typical ransomware attack, the malicious process renames the user files to make it more difficult
        for the users to identify and recover their files. Our system increases the malice score of a
        process as the process performs file rename operations. In fact, a process is assigned a higher
        malice score if it renames all the files in a directory.
        """
        if r['major'] == 'IRP_MJ_SET_INFORMATION' and r['fileinfoclass'] == 'FileRenameInformation':
            self.file_dimensions[r['newfilename']] = self.file_dimensions[r['dest_path']]
            del self.file_dimensions[r['dest_path']]

    def update_file_sequential_writes(self, r: dict):
        """
        Sequential Writes. Redemption monitors the sequential write operations performed by a process. In a
        typical ransomware attack, the malicious process writes to the user files sequentially to encrypt
        them. Our system increases the malice score of a process as the process performs sequential write
        operations. In fact, a process is assigned a higher malice score if it writes sequentially to all
        the content of the files.
        """
        # scoperta dei path
        if r['major'] == "IRP_MJ_DIRECTORY_CONTROL" and r["minor"] == 'IRP_MN_QUERY_DIRECTORY' and r['dest_path'] not in self.discovered_paths:
            self.discovered_paths.add(r['dest_path'])

        if r['major'] == 'IRP_MJ_WRITE':
            # se il file è in uno dei path scoperti scopri quale con le regex e aggiorna le operazioni di scrittura su quel path
            # se r['dest_path'] - ultima parte del path è in discovered_paths allora aggiorna le operazioni di scrittura su quel path
            parsed_path = "".join(r['dest_path'].split('\\')[:-2]) # tutto tranne l'ultima parte del path
            if parsed_path in self.discovered_paths:
                if parsed_path not in self.file_writes:
                    self.file_writes[parsed_path] = set()
                self.file_writes[parsed_path].add(r['dest_path'])

    def file_sequential_writes(self, r: dict):
        """
        Redemption monitors if the process traverses over the user files with write
        privilege, and computes the additive inverse of the number of privileged accesses
        to unique files in a given path. The output of the function resides on [0,1]. Given
        a process X, the function assigns a higher malice score as X generates more write
        requests to access files in a given path. Here, write(X, fi) is the ith independent
        write request generated by the process X on a given file fi.
        """

        # self.path_writes = {} # chiave: path, valore: set di file unici scritti su quel path (identificati dal file name)
        # formula: result = 1 - 1/(1 + number of unique files written on that path)
        parsed_path = "".join(r['dest_path'].split('\\')[:-2]) # tutto tranne l'ultima parte del path
        if parsed_path in self.file_writes:
            return 1 - 1/(1 + len(self.file_writes[parsed_path]))
        return 0

    def update_extension_writes(self, r: dict):
        """
        Extension of File Writes. Redemption monitors the file extensions of the user files that a process writes to. In a typical ransomware attack, the malicious process writes to user files with specific extensions (e.g., .doc, .xls, .pdf) to encrypt them. Our system increases the malice score of a process as the process writes to user files with different extensions. In fact, a process is assigned a higher malice score if it writes to user files with many different extensions.
        """
        if r['major'] == 'IRP_MJ_WRITE':
            extension = r['dest_path'].split('.')[-1]
            if extension not in self.extension_writes:
                self.extension_writes[extension] = 0
            if len(self.extension_writes) < 2: # se abbiamo già scritto su 2 estensioni diverse non ci interessa più aggiornare questa feature
                self.extension_writes.append(extension)
    
    def extension_writes(self, r: dict):
        """
        Given a set of document classes, Redemption monitors whether the pro-
        cess requests write access to files that belong to different document classes. The
        file A and file B belong to two different document classes if the program that
        opens file A cannot take file B as a valid input. For example, a docx and a
        pdf file belong to two different document classes since a docx file cannot be
        opened via a PDF editor program. We assign the score 1 if the process performs
        cross-document access requests similar to ransomware.
        """
        # formula: 
        return 1 if len(self.extension_writes) > 1 else 0

    def dt(self, r: dict):
        """
        calcoliamo il tempo trascorso tra l'ultima operazione e r
        """
        if self.last_time is None:
            self.last_time = r['time']
            return 0
        else:
            dt = r['time'] - self.last_time
            self.last_time = r['time']
            return 1/dt if dt > 0 else 0