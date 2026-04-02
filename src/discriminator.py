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

        # Dropout applicato all'input della LSTM per regolarizzazione
        # Riduce l'overfitting sul training set durante la fase avversaria
        self.input_dropout = nn.Dropout(0.1)

        # Livello LSTM (bidirezionale)
        # Prende in input la concatenazione del token e dell'etichetta e produce una rappresentazione della sequenza.
        # bidirectional=True permette alla LSTM di leggere il payload in entrambe le direzioni, migliorando la comprensione del contesto.
        self.lstm = nn.LSTM(embed_dim * 2, hidden_dim, batch_first=True, bidirectional=True)

        # Dropout applicato all'output della LSTM
        # Previene la co-adattazione tra neuroni e migliora la generalizzazione
        self.lstm_dropout = nn.Dropout(0.3)

        # Classificatore a due livelli per aumentare la capacità discriminativa:
        # - fc1: proietta l'output BiLSTM su uno spazio intermedio con attivazione non-lineare
        # - fc2: riduce a un singolo logit (reale/falso)
        # LeakyReLU evita i "dying neurons" consentendo un piccolo gradiente anche per valori negativi
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(hidden_dim, 1)

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

        # Concatenazione del token con l'etichetta e dropout sull'input
        lstm_input = torch.cat([emb_x, emb_label], dim=2)  # (batch_size, seq_len, embed_dim * 2)
        lstm_input = self.input_dropout(lstm_input)

        # Processiamo la sequenza con la LSTM
        # out contiene l'output per ogni step temporale.
        out, (hidden, cell) = self.lstm(lstm_input)  # out: (batch_size, seq_len, hidden_dim * 2)

        # Estraiamo il significato dell'intera frase
        # Poiché è bidirezionale, concateniamo l'ultimo stato avanti e indietro
        hidden_forward = hidden[-2]  # Ultimo stato della direzione "avanti"
        hidden_backward = hidden[-1]  # Ultimo stato della direzione "indietro"
        final_state = torch.cat((hidden_forward, hidden_backward), dim=1)  # (batch_size, hidden_dim * 2)
        final_state = self.lstm_dropout(final_state)

        # Classificatore a due strati: proiezione intermedia → logit finale
        h = self.leaky_relu(self.fc1(final_state))  # (batch_size, hidden_dim)
        logits = self.fc2(h)                         # (batch_size, 1)

        # Nota: restituiamo i "logits" (numeri grezzi). 
        # La conversione in probabilità (0-1) con la funzione Sigmoide avverrà 
        # durante il calcolo della loss (usando BCEWithLogitsLoss) per maggiore stabilità numerica.
        return logits.squeeze(-1)  # Rimuoviamo dimensioni extra, forma finale: (batch_size)