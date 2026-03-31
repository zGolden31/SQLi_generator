import torch
import torch.nn as nn
import torch.optim as optim

# Import dei modelli Generator e Discriminator
from generator import ConditionalGenerator
from discriminator import ConditionalDiscriminator

def train_cgan():
    # --- Configurazione iperparametri ---
    vocab_size = 379  # Dimensione del vocabolario (numero di token unici)
    embed_dim = 64    # Dimensione dei vettori di embedding
    hidden_dim = 128  # Dimensione dello stato nascosto della LSTM
    num_classes = 3   # Numero di classi per le etichette condizionali (es. 'ok', 'error', 'fail')

    batch_size = 32
    seq_len = 50 # Lunghezza massima dei payload SQL (può essere adattata in base ai dati)
    epochs = 50

    # --- Inizializzazione dei modelli ---
    generator = ConditionalGenerator(vocab_size, embed_dim, hidden_dim, num_classes)
    discriminator = ConditionalDiscriminator(vocab_size, embed_dim, hidden_dim, num_classes)

    # Usiamo la GPU se disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # --- Ottimizzatori ---
    g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

    # Loss function
    # Per il pre-training del generatore (prevedere il token esatto)
    g_pretrain_loss = nn.CrossEntropyLoss()
    # Per il discriminatore (reale vs falso) sarà binario
    d_loss_fn = nn.BCEWithLogitsLoss()

    # --- SIMULAZIONE DATI ---
    # TODO: Sostituire con il caricamento dei dati reali
    # Per ora, generiamo dati casuali per testare il processo di training
    def get_real_batch():
        # Dati casuali finti: (batch_size, seq_len)
        real_data = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        labels = torch.randint(0, num_classes, (batch_size,)).to(device)
        return real_data, labels
    print("Inizio del pre-training del generatore...")
    for step in range(100):
        real_data, labels = get_real_batch()

        g_optimizer.zero_grad()
        # Per prevedere il token successivo, passiamo tutti i token tranne l'ultimo (input)
        # e cerchiamo di prevedere tutti i token tranne il primo (target)
        inputs = real_data[:, :-1]  # (batch_size, seq_len - 1)
        targets = real_data[:, 1:]  # (batch_size, seq_len - 1)

        logits, _ = generator(inputs, labels)

        # Rimodelliamo per il calcolo della loss
        loss = g_pretrain_loss(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        g_optimizer.step()

    print("Pre-training del discriminatore...")
    for step in range(100):
        real_data, labels = get_real_batch()

        # --- Addestramento del Discriminatore ---
        d_optimizer.zero_grad()

        # Dati reali
        real_labels = torch.ones(batch_size, 1).to(device)  # Etichetta 1 per dati reali
        real_targets = torch.ones_like(real_logits) # 1 = Reale
        loss_real = d_loss_fn(real_logits, real_targets)

        # Dati generati
        fake_data = generator.sample(batch_size, start_token_id=2, labels=labels, max_seq_len=seq_len)
        fake_logits = discriminator(fake_data.detach(), labels) # detach() stacca i gradienti del generatore
        fake_targets = torch.zeros_like(fake_logits) # 0 = Falso
        loss_fake = d_loss_fn(fake_logits, fake_targets)
        
        # Loss totale del discriminatore
        loss_d = loss_real + loss_fake
        loss_d.backward()
        d_optimizer.step()

    print("Inizio Addestramento Avversario (Reinforcement Learning)...")
    for epoch in range(epochs):
        
        # --- A. Aggiorniamo il Generatore (Policy Gradient) ---
        g_optimizer.zero_grad()
        _, labels = get_real_batch()
        
        # Il generatore crea un payload
        fake_data = generator.sample(batch_size, start_token_id=2, labels=labels, max_seq_len=seq_len)
        
        # Il discriminatore lo valuta (ci dà il "Reward" o Ricompensa)
        # Sigmoid trasforma i logits grezzi in una percentuale da 0 a 1 (1 = Discriminatore Ingannato)
        reward = torch.sigmoid(discriminator(fake_data, labels)).detach() 
        
        # Per calcolare la loss del generatore col Reinforcement Learning (REINFORCE),
        # ci servono le probabilità che il generatore aveva assegnato ai token che ha scelto.
        logits, _ = generator(fake_data, labels) 
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Estraiamo le probabilità solo dei token effettivamente campionati
        chosen_log_probs = log_probs.gather(2, fake_data.unsqueeze(2)).squeeze(2)
        
        # Formula semplificata di Policy Gradient: Loss = - (Ricompensa * Probabilità delle azioni)
        # Se la ricompensa è alta, la loss diventa molto negativa, incoraggiando queste azioni.
        # Calcoliamo la media su tutta la frase e su tutto il batch
        g_loss = - (reward.unsqueeze(1) * chosen_log_probs).mean()
        
        g_loss.backward()
        g_optimizer.step()
        
        # --- B. Aggiorniamo il Discriminatore ---
        d_optimizer.zero_grad()
        real_data, _ = get_real_batch()
        
        # Ricalcoliamo le predizioni sui nuovi payload
        real_logits = discriminator(real_data, labels)
        fake_logits = discriminator(fake_data.detach(), labels)
        
        loss_d = d_loss_fn(real_logits, torch.ones_like(real_logits)) + \
                 d_loss_fn(fake_logits, torch.zeros_like(fake_logits))
        
        loss_d.backward()
        d_optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | G Loss: {g_loss.item():.4f} | D Loss: {loss_d.item():.4f}")

if __name__ == "__main__":
    train_cgan()