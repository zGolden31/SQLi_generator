import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        """
        Inizializza il generatore condizionato.

        Parametri:
        - vocab_size: dimensione del vocabolario (dal JSON per adesso è circa 379)
        - embed_dim: dimensione del vettore in trasformeremo i token (circa 32 o 64)
        - hidden_dim: La dimensione della memoria nascosta dell'LSTM (es. 64 o 128).
        - num_classes: Il numero di etichette condizionali (es. 3, per 'ok', 'error', 'fail').
        """
        # Parametri di configurazione del modello
        super(ConditionalGenerator, self).__init__()
        self.hidden_dim = hidden_dim

        # Livello di Embedding per i Token SQL
        # Trasforma gli ID interi del dizionario BPE in vettori densi di numeri.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Livello di Embedding per le Etichette (Condizione)
        # Trasforma le etichette condizionali (es. 'ok', 'error', 'fail') in un vettore della stessa dimensione.
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Livello LSTM 
        # Prende in input la concatenazione del token e della condizione
        # La dimensione di ingresso è embed_dim (token) + embed_dim (label) = embed_dim * 2
        self.lstm = nn.LSTM(embed_dim * 2, hidden_dim, batch_first=True)

        # Livello lineare (Output)
        # Mappa l'output della LSTM (hidden_dim) alla dimensione del vocabolario
        # Serve per calcolare la probabiltà del token successivo da generare.
        self.fc = nn.Linear(hidden_dim, vocab_size)
