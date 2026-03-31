import torch
import torch.nn as nn

class ConditionalDiscriminator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        """
        Inizializzazione del modello Conditional Discriminator.

        Parametri:
        - vocab_size: La dimensione del vocabolario (numero totale di token unici).
        - embed_dim: La dimensione dei vettori di embedding per i token e le etichette.
        - hidden_dim: La dimensione dello stato nascosto della LSTM.
        - num_classes: Il numero di classi per le etichette condizionali (tipologie di SQLi o risultati).
        """

        super(ConditionalDiscriminator, self).__init__()

        # Livelli di embedding 
        # Trasformano gli ID dei token e delle etichette in vettori densi.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Livello LSTM (bidirezionale)
        # Prende in input la concatenazione del token e dell'etichetta e produce una rappresentazione della sequenza.
        # bidirectional=True permette alla LSTM di leggere il payload in entrambe le direzioni, migliorando la comprensione del contesto.
        self.lstm = nn.LSTM(embed_dim * 2, hidden_dim, batch_first=True, bidirectional=True)

        # Livello lineare (Classificatore finale)
        # Dato che la LSTM è bidirezionale, l'output ha dimensione hidden_dim * 2 (per i due direzioni).
        # Vogliamo riduerre questo output ad un sincolo numero (Reale o Falso)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, labels):
        """
        Passaggio in avanti. 
        Valuta se un batch di payload è reale o generato.

        Parametri:
        - x: Tensore dei token di input (batch_size, seq_len)
        - labels: Tensore delle etichette condizionali (batch_size, 1)
        """
        batch_size, seq_len = x.size()

        # Convertiamo i token in vettori
        emb_x = self.token_embedding(x)  # (batch_size, seq_len, embed_dim)

        # Convertiamo le etichette in vettori
        # Assicuriamoci che labels abbia la forma giusta e convertiamolo in embedding
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)  # (batch_size, 1)
        emb_label = self.label_embedding(labels)  # (batch_size, 1, embed_dim)

        # Espandiamo l'etichetta per tutta la lunghezza della sequenza
        # Vogliamo ricordare alla rete la condizione di ogni singolo token letto
        emb_label = emb_label.expand(batch_size, seq_len, -1)  # (batch_size, seq_len, embed_dim)

        # Concatenazione del token con l'etichetta
        lstm_input = torch.cat([emb_x, emb_label], dim=2)  # (batch_size, seq_len, embed_dim * 2)

        # Processiamo la sequenza con la LSTM
        # out contiene l'output per ogni step temporale.
        out, (hidden, cell) = self.lstm(lstm_input)  # out: (batch_size, seq_len, hidden_dim * 2)

        # Estraiamo il significato dell'intera frase
        # Invece di guardare tutti i token, prendiamo l'output della LSTM relativo
        # all'ultimo step della sequenza, che riassume tutto il payload.
        # Poiché è bidirezionale, prendiamo l'ultimo stato della direzione "avanti" e "indietro"
        # Per semplicità, possiamo semplicemente fare la media o prendere l'ultimo elemento di 'out'
        # out[:, -1, :] prende l'ultimo step dell'intera sequenza per ogni elemento del batch.
        final_state = out[:, -1, :] # Forma: (batch_size, hidden_dim * 2)

        # Calcoliamo il punteggio finale (probabilità che il payload sia reale)
        logits = self.fc(final_state)  # (batch_size, 1)

        # Nota: restituiamo i "logits" (numeri grezzi). 
        # La conversione in probabilità (0-1) con la funzione Sigmoide avverrà 
        # durante il calcolo della loss (usando BCEWithLogitsLoss) per maggiore stabilità numerica.
        return logits.squeeze(-1)  # Rimuoviamo dimensioni extra, forma finale: (batch_size)