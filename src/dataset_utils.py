import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tokenizers import Tokenizer
from config import PAD_TOKEN, START_TOKEN

class SQLiDataset(Dataset):
    def __init__(self, csv_file, tokenizer_config_path, max_seq_len=50, label_value=0):
        """
        Inizializza il Dataset per leggere i payload SQLi.
        
        Parametri:
        - csv_file: Il percorso del file CSV (es. 'data/raw/error_based.csv')
        - tokenizer_config_path: Il percorso del tuo file JSON BPE.
        - max_seq_len: La lunghezza fissa di ogni payload (es. 50 token).
        - label_value: L'etichetta associata a questo file (es. 0 per error_based).
        """
        # 1. Carichiamo il Tokenizer BPE che hai già addestrato
        self.tokenizer = Tokenizer.from_file(tokenizer_config_path)
        
        # Salviamo i parametri
        self.max_seq_len = max_seq_len
        self.label_value = label_value
        
        # 2. Leggiamo il file CSV
        # Usiamo pandas come hai fatto nel tuo file tokenizator.py
        df = pd.read_csv(csv_file)
        
        # Assicuriamoci che la colonna esista (adatta il nome se necessario)
        col_name = 'request/payload' if 'request/payload' in df.columns else 'payload'
        
        # Estraiamo tutti i payload come lista di stringhe, ignorando i valori vuoti
        self.payloads = df[col_name].dropna().astype(str).tolist()

    def __len__(self):
        """Restituisce il numero totale di payload nel dataset."""
        return len(self.payloads)

    def __getitem__(self, idx):
        """
        Prende un singolo payload, lo converte in numeri e lo formatta.
        Questo metodo viene chiamato automaticamente dal DataLoader di PyTorch.
        """
        testo = self.payloads[idx]
        
        # 1. Convertiamo il testo in token ID usando il BPE
        # encode() trasforma la stringa in una sequenza di numeri
        encoded = self.tokenizer.encode(testo)
        token_ids = encoded.ids
        
        # 2. Aggiungiamo il token [CLS] all'inizio (ID 2 nel tuo JSON)
        # Questo serve al Generatore per capire dove inizia la frase
        token_ids = [START_TOKEN] + token_ids
        
        # 3. Troncamento o Padding
        # Le reti neurali vogliono sequenze tutte della stessa lunghezza!
        if len(token_ids) > self.max_seq_len:
            # Se è troppo lungo, lo tagliamo
            token_ids = token_ids[:self.max_seq_len]
        else:
            # Se è troppo corto, aggiungiamo il token [PAD] (ID 0 nel tuo JSON) alla fine
            padding_length = self.max_seq_len - len(token_ids)
            token_ids = token_ids + [PAD_TOKEN] * padding_length
            
        # 4. Convertiamo in Tensori PyTorch
        x_tensor = torch.tensor(token_ids, dtype=torch.long)
        label_tensor = torch.tensor(self.label_value, dtype=torch.long)
        
        return x_tensor, label_tensor

def get_dataloader(csv_file, tokenizer_config_path, batch_size, max_seq_len, label_value):
    """
    Funzione di comodità per creare direttamente il DataLoader.
    """
    dataset = SQLiDataset(csv_file, tokenizer_config_path, max_seq_len, label_value)
    
    # Il DataLoader si occuperà di mescolare i dati (shuffle=True) 
    # e dividerli in batch (es. 32 payload alla volta).
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader