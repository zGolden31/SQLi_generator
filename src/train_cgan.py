import torch
import torch.nn as nn
import torch.optim as optim

# Import dei modelli Generator e Discriminator
from generator import ConditionalGenerator
from discriminator import ConditionalDiscriminator
from dataset_utils import get_dataloader

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
    g_pretrain_loss = nn.CrossEntropyLoss(ignore_index=0)
    # Per il discriminatore (reale vs falso) sarà binario
    d_loss_fn = nn.BCEWithLogitsLoss()

    # Caricamento dati
    csv_file = "data/raw/error_based.csv"
    tokenizer_json = "data/bpe_config/sql_bpe_tokenizer_config.json"

    print("Caricamento dataset in corso...")
    dataloader = get_dataloader(
        csv_file=csv_file, 
        tokenizer_config_path=tokenizer_json, 
        batch_size=batch_size, 
        max_seq_len=seq_len, 
        label_value=0  # 0 rappresenta 'error_based'
    )

    print("Inizio del pre-training del generatore...")
    # Facciamo 5 "giri" (epoche) su tutto il dataset reale per fargli imparare la base
    for pre_epoch in range(5): 
        for real_data, labels in dataloader:
            # Spostiamo i dati sulla GPU (se disponibile)
            real_data, labels = real_data.to(device), labels.to(device)

            g_optimizer.zero_grad()
            
            # Input (dal primo al penultimo) e Target (dal secondo all'ultimo)
            inputs = real_data[:, :-1]  
            targets = real_data[:, 1:]  

            logits, _ = generator(inputs, labels)

            loss = g_pretrain_loss(logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            g_optimizer.step()
        print(f"Pre-training G - Epoca {pre_epoch+1}/5 completata.")

    print("Inizio del pre-training del discriminatore...")
    for pre_epoch in range(5):
        for real_data, labels in dataloader:
            real_data, labels = real_data.to(device), labels.to(device)

            d_optimizer.zero_grad()

            # --- Dati reali ---
            # QUI IL FIX: Prima calcoliamo le predizioni sui dati veri!
            real_logits = discriminator(real_data, labels) 
            real_targets = torch.ones_like(real_logits) # 1 = Reale
            loss_real = d_loss_fn(real_logits, real_targets)

            # --- Dati generati ---
            fake_data = generator.sample(batch_size, start_token_id=2, labels=labels, max_seq_len=seq_len)
            fake_logits = discriminator(fake_data.detach(), labels) 
            fake_targets = torch.zeros_like(fake_logits) # 0 = Falso
            loss_fake = d_loss_fn(fake_logits, fake_targets)
            
            # Loss totale
            loss_d = loss_real + loss_fake
            loss_d.backward()
            d_optimizer.step()
        print(f"Pre-training D - Epoca {pre_epoch+1}/5 completata.")

    print("Inizio Addestramento Avversario (Reinforcement Learning)...")
    for epoch in range(epochs):
        # Invece di un solo batch casuale, passiamo attraverso tutti i batch reali
        for batch_idx, (real_data, labels) in enumerate(dataloader):
            real_data, labels = real_data.to(device), labels.to(device)
            
            # --- A. Aggiorniamo il Generatore (Policy Gradient) ---
            g_optimizer.zero_grad()
            
            fake_data = generator.sample(batch_size, start_token_id=2, labels=labels, max_seq_len=seq_len)
            reward = torch.sigmoid(discriminator(fake_data, labels)).detach() 
            
            logits, _ = generator(fake_data, labels) 
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            chosen_log_probs = log_probs.gather(2, fake_data.unsqueeze(2)).squeeze(2)
            
            g_loss = - (reward.unsqueeze(1) * chosen_log_probs).mean()
            
            g_loss.backward()
            g_optimizer.step()
            
            # --- B. Aggiorniamo il Discriminatore ---
            d_optimizer.zero_grad()
            
            real_logits = discriminator(real_data, labels)
            fake_logits = discriminator(fake_data.detach(), labels)
            
            loss_d = d_loss_fn(real_logits, torch.ones_like(real_logits)) + \
                     d_loss_fn(fake_logits, torch.zeros_like(fake_logits))
            
            loss_d.backward()
            d_optimizer.step()
            
        # Stampiamo i risultati solo alla fine di ogni intera "epoca" (non per ogni batch)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs} | G Loss: {g_loss.item():.4f} | D Loss: {loss_d.item():.4f}")

        # Salvataggio dei modelli
    
    print("Salvataggio dei modelli in corso...")
    torch.save(generator.state_dict(), "data/models/generator_model.pth")
    torch.save(discriminator.state_dict(), "data/models/discriminator_model.pth")
    print("Addestramento completato e modelli salvati!")


if __name__ == "__main__":
    train_cgan()