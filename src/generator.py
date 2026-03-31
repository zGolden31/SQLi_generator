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

    def forward(self, x, labels, hidden=None):
        """
        Passaggio in avanti (forward pass) del modello.
        Usato durante il pre-training Maximum Likelihood Estimation (MLE).

        Parametri:
        - x: Tensore dei token di input (batch_size, seq_len) - sequenza di token SQL codificati come ID interi
        - labels: Tensore delle etichette condizionali (batch_size, 1) - ad esempio, 0 per 'ok', 1 per 'error', 2 per 'fail'
        - hidden: Stato nascosto iniziale per la LSTM 
        """
        batch_size, seq_len = x.size()

        # Otteniamo la rappresentazione vettoriale dei token SQL
        emb_x = self.token_embedding(x)  # (batch_size, seq_len, embed_dim)

        # Otteniamo la rappresentazione vettoriale delle etichette condizionali
        emb_label = self.label_embedding(labels)  # (batch_size, 1, embed_dim)

        # Espandiamo l'etichetta per tutta la lunghezza della sequenza
        # Vogliamo che l'LSTM ricordi la condizione ad ogni singolo token che processa
        emb_label = emb_label.expand(batch_size, seq_len, -1)  # (batch_size, seq_len, embed_dim)

        # Concatenazione del token con l'etichetta
        lstm_input = torch.cat([emb_x,emb_label], dim=2)  # (batch_size, seq_len, embed_dim * 2)

        # Passare i dati nell'LSTM
        out, hidden = self.lstm(lstm_input, hidden)  # out: (batch_size, seq_len, hidden_dim)

        # Otteniamo i punteggi finali per ogni token del vocabolario
        logits = self.fc(out)  # (batch_size, seq_len, vocab_size)

        return logits, hidden
    
    def sample(self, batch_size, start_token_id, labels, max_seq_len):
        """
        Metodo per generare (campionare) nuovi payload da zero.
        Il generatore parte dal token iniziale e genera un token alla volta.
        """
        # Creazione del primo input: una colonna piena del token iniziale
        inputs = torch.LongTensor(batch_size, 1).fill_(start_token_id).to(labels.device)  # (batch_size, 1)

        # Se c'è una GPU disponibile, spostiamo il modello e i tensori su di essa
        if next(self.parameters()).is_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        hidden = None # Stato nascosto iniziale
        samples = [] # Lista per memorizzare i token generati

        for _ in range(max_seq_len):
            # Passiamo il token attuale e l'etichetta attraverso la rete
            logits, hidden = self.forward(inputs, labels, hidden)  # logits: (batch_size, 1, vocab_size)

            # Estraiamo l'output per l'ultimo token calcolato
            logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # Convertiamo i punteggi in probabilità usando softmax
            probs = F.softmax(logits, dim=-1)  # (batch_size, vocab_size)

            # Campioniamo il token successivo in base alle probabilità
            next_token = torch.multinomial(probs, 1)  # (batch_size, 1)   

            # Aggiungiamo il token campionato alla nostra lista
            samples.append(next_token)

            # Il token campionato diventa l'input per il prossimo passo
            inputs = next_token

        # uniamo tutti i token generati in un unico tensore
        samples = torch.cat(samples, dim=1)  # (batch_size, max_seq_len)
        return samples    