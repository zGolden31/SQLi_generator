import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path

# Permette l'esecuzione diretta del file (python src/training/train_cgan.py)
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES,
    GENERATOR_MODEL, DISCRIMINATOR_MODEL, TOKENIZER_CONFIG,
    CSV_FILE, BATCH_SIZE, MAX_SEQ_LEN, EPOCHS, PRETRAIN_EPOCHS,
    LEARNING_RATE_G, LEARNING_RATE_D, START_TOKEN, GRAD_CLIP
)

# Import dei modelli Generator e Discriminator
from generator import ConditionalGenerator
from discriminator import ConditionalDiscriminator
from dataset_loader import get_dataloader

def train_cgan():
    # --- Inizializzazione dei modelli ---
    generator = ConditionalGenerator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
    discriminator = ConditionalDiscriminator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)

    # Usiamo la GPU se disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # --- Ottimizzatori ---
    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D)

    # Loss function
    # Per il pre-training del generatore (prevedere il token esatto)
    g_pretrain_loss = nn.CrossEntropyLoss(ignore_index=0)
    # Per il discriminatore (reale vs falso) sarà binario
    d_loss_fn = nn.BCEWithLogitsLoss()

    print("Caricamento dataset in corso...")
    dataloader = get_dataloader(
        csv_file=CSV_FILE, 
        tokenizer_config_path=TOKENIZER_CONFIG, 
        batch_size=BATCH_SIZE, 
        max_seq_len=MAX_SEQ_LEN, 
        label_value=0  # 0 rappresenta 'error_based'
    )

    print("Inizio del pre-training del generatore...")
    # Facciamo "giri" (epoche) su tutto il dataset reale per fargli imparare la base
    for pre_epoch in range(PRETRAIN_EPOCHS): 
        for real_data, labels in dataloader:
            # Spostiamo i dati sulla GPU (se disponibile)
            real_data, labels = real_data.to(device), labels.to(device)

            g_optimizer.zero_grad()
            
            # Input (dal primo al penultimo) e Target (dal secondo all'ultimo)
            inputs = real_data[:, :-1]  
            targets = real_data[:, 1:]  

            logits, _ = generator(inputs, labels)

            loss = g_pretrain_loss(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP)  # Clipping dei gradienti

            g_optimizer.step()
        print(f"Pre-training G - Epoca {pre_epoch+1}/{PRETRAIN_EPOCHS} completata.")

    print("Inizio del pre-training del discriminatore...")
    for pre_epoch in range(PRETRAIN_EPOCHS):
        for real_data, labels in dataloader:
            real_data, labels = real_data.to(device), labels.to(device)

            d_optimizer.zero_grad()

            # --- Dati reali ---
            # QUI IL FIX: Prima calcoliamo le predizioni sui dati veri!
            real_logits = discriminator(real_data, labels) 
            real_targets = torch.ones_like(real_logits) # 1 = Reale
            loss_real = d_loss_fn(real_logits, real_targets)

            # --- Dati generati ---
            fake_data = generator.sample(batch_size=BATCH_SIZE, start_token_id=START_TOKEN, labels=labels, max_seq_len=MAX_SEQ_LEN)
            fake_logits = discriminator(fake_data.detach(), labels) 
            fake_targets = torch.zeros_like(fake_logits) # 0 = Falso
            loss_fake = d_loss_fn(fake_logits, fake_targets)
            
            # Loss totale
            loss_d = loss_real + loss_fake
            loss_d.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP)  # Clipping dei gradienti
            d_optimizer.step()
        print(f"Pre-training D - Epoca {pre_epoch+1}/{PRETRAIN_EPOCHS} completata.")

    print("Inizio Addestramento Avversario (Reinforcement Learning)...")
    for epoch in range(EPOCHS):
        # Invece di un solo batch casuale, passiamo attraverso tutti i batch reali
        for batch_idx, (real_data, labels) in enumerate(dataloader):
            real_data, labels = real_data.to(device), labels.to(device)
            
            # --- A. Aggiorniamo il Generatore (Policy Gradient) ---
            g_optimizer.zero_grad()
            
            fake_data = generator.sample(batch_size=BATCH_SIZE, start_token_id=START_TOKEN, labels=labels, max_seq_len=MAX_SEQ_LEN)
            reward = torch.sigmoid(discriminator(fake_data, labels)).detach() 
            
            # Fix per il gradiente
            # Ricreiamo gli input: [CLS] + tutti i fake_data tranne l'ultimo
            start_tokens = torch.LongTensor(BATCH_SIZE, 1).fill_(START_TOKEN).to(device)
            pg_inputs = torch.cat([start_tokens, fake_data[:, :-1]], dim=1)

            # Ora calcoliamo le probabilità: pg_inputs e fake_data (target) sono allineati
            logits, _ = generator(pg_inputs, labels)
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
            nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP)  # Clipping dei gradienti
            d_optimizer.step()
            
        # Stampiamo i risultati solo alla fine di ogni intera "epoca" (non per ogni batch)
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch}/{EPOCHS} | G Loss: {g_loss.item():.4f} | D Loss: {loss_d.item():.4f}")

        # Salvataggio dei modelli
    
    print("Salvataggio dei modelli in corso...")
    torch.save(generator.state_dict(), GENERATOR_MODEL)
    torch.save(discriminator.state_dict(), DISCRIMINATOR_MODEL)
    print("Addestramento completato e modelli salvati!")


if __name__ == "__main__":
    train_cgan()