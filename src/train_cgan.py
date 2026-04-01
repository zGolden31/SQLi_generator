import csv                                                                    # [LOG]
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path                                                      # [LOG]

from config import (
    VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES,
    GENERATOR_MODEL, DISCRIMINATOR_MODEL, TOKENIZER_CONFIG,
    CSV_FILE, BATCH_SIZE, MAX_SEQ_LEN, EPOCHS, PRETRAIN_EPOCHS,
    LEARNING_RATE_G, LEARNING_RATE_D, START_TOKEN, GRAD_CLIP,
    MODELS_DIR                                                                # [LOG]
)

from generator import ConditionalGenerator
from discriminator import ConditionalDiscriminator
from dataset_loader import get_dataloader

def train_cgan():
    # --- Inizializzazione dei modelli ---
    generator = ConditionalGenerator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
    discriminator = ConditionalDiscriminator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # --- Ottimizzatori ---
    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D)

    g_pretrain_loss = nn.CrossEntropyLoss(ignore_index=0)
    d_loss_fn       = nn.BCEWithLogitsLoss()

    print("Caricamento dataset in corso...")
    dataloader = get_dataloader(
        csv_file=CSV_FILE,
        tokenizer_config_path=TOKENIZER_CONFIG,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        label_value=0
    )

    # --- Apertura file di log ---                                            # [LOG]
    log_path = Path(MODELS_DIR) / "training_log.csv"                         # [LOG]
    log_file = open(log_path, "w", newline="", encoding="utf-8")             # [LOG]
    log_writer = csv.writer(log_file)                                         # [LOG]
    log_writer.writerow([                                                     # [LOG]
        "phase", "epoch",                                                     # [LOG]
        "g_loss", "d_loss",                                                   # [LOG]
        "reward_mean", "d_real_mean", "d_fake_mean"                          # [LOG]
    ])                                                                        # [LOG]

    print("Inizio del pre-training del generatore...")
    for pre_epoch in range(PRETRAIN_EPOCHS):
        epoch_g_loss = 0.0; n_batches = 0                                    # [LOG]
        for real_data, labels in dataloader:
            real_data, labels = real_data.to(device), labels.to(device)

            g_optimizer.zero_grad()
            inputs  = real_data[:, :-1]
            targets = real_data[:, 1:]
            logits, _ = generator(inputs, labels)
            loss = g_pretrain_loss(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP)
            g_optimizer.step()

            epoch_g_loss += loss.item(); n_batches += 1                      # [LOG]

        avg_g = epoch_g_loss / max(n_batches, 1)                             # [LOG]
        log_writer.writerow(["pretrain_g", pre_epoch+1, f"{avg_g:.6f}", "", "", "", ""])  # [LOG]
        log_file.flush()                                                      # [LOG]
        print(f"Pre-training G - Epoca {pre_epoch+1}/{PRETRAIN_EPOCHS} completata.")

    print("Inizio del pre-training del discriminatore...")
    for pre_epoch in range(PRETRAIN_EPOCHS):
        epoch_d_loss = 0.0; epoch_d_real = 0.0; epoch_d_fake = 0.0; n_batches = 0  # [LOG]
        for real_data, labels in dataloader:
            real_data, labels = real_data.to(device), labels.to(device)

            d_optimizer.zero_grad()

            real_logits  = discriminator(real_data, labels)
            real_targets = torch.ones_like(real_logits)
            loss_real    = d_loss_fn(real_logits, real_targets)

            fake_data   = generator.sample(batch_size=BATCH_SIZE, start_token_id=START_TOKEN, labels=labels, max_seq_len=MAX_SEQ_LEN)
            fake_logits = discriminator(fake_data.detach(), labels)
            fake_targets = torch.zeros_like(fake_logits)
            loss_fake   = d_loss_fn(fake_logits, fake_targets)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP)
            d_optimizer.step()

            epoch_d_loss += loss_d.item()                                    # [LOG]
            epoch_d_real += torch.sigmoid(real_logits).mean().item()        # [LOG]
            epoch_d_fake += torch.sigmoid(fake_logits).mean().item()        # [LOG]
            n_batches    += 1                                                 # [LOG]

        avg_d      = epoch_d_loss / max(n_batches, 1)                        # [LOG]
        avg_d_real = epoch_d_real / max(n_batches, 1)                        # [LOG]
        avg_d_fake = epoch_d_fake / max(n_batches, 1)                        # [LOG]
        log_writer.writerow(["pretrain_d", pre_epoch+1, "", f"{avg_d:.6f}", "", f"{avg_d_real:.4f}", f"{avg_d_fake:.4f}"])  # [LOG]
        log_file.flush()                                                      # [LOG]
        print(f"Pre-training D - Epoca {pre_epoch+1}/{PRETRAIN_EPOCHS} completata.")

    print("Inizio Addestramento Avversario (Reinforcement Learning)...")
    for epoch in range(EPOCHS):
        epoch_g_loss = 0.0; epoch_d_loss = 0.0                               # [LOG]
        epoch_reward = 0.0; epoch_d_real = 0.0; epoch_d_fake = 0.0          # [LOG]
        n_batches = 0                                                         # [LOG]

        for batch_idx, (real_data, labels) in enumerate(dataloader):
            real_data, labels = real_data.to(device), labels.to(device)

            # --- A. Aggiorniamo il Generatore (Policy Gradient) ---
            g_optimizer.zero_grad()

            fake_data = generator.sample(batch_size=BATCH_SIZE, start_token_id=START_TOKEN, labels=labels, max_seq_len=MAX_SEQ_LEN)
            reward = torch.sigmoid(discriminator(fake_data, labels)).detach()

            start_tokens = torch.LongTensor(BATCH_SIZE, 1).fill_(START_TOKEN).to(device)
            pg_inputs = torch.cat([start_tokens, fake_data[:, :-1]], dim=1)

            logits, _ = generator(pg_inputs, labels)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            chosen_log_probs = log_probs.gather(2, fake_data.unsqueeze(2)).squeeze(2)

            g_loss = -(reward.unsqueeze(1) * chosen_log_probs).mean()
            g_loss.backward()
            g_optimizer.step()

            # --- B. Aggiorniamo il Discriminatore ---
            d_optimizer.zero_grad()

            real_logits = discriminator(real_data, labels)
            fake_logits = discriminator(fake_data.detach(), labels)

            loss_d = d_loss_fn(real_logits, torch.ones_like(real_logits)) + \
                     d_loss_fn(fake_logits, torch.zeros_like(fake_logits))
            loss_d.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP)
            d_optimizer.step()

            epoch_g_loss += g_loss.item()                                    # [LOG]
            epoch_d_loss += loss_d.item()                                    # [LOG]
            epoch_reward += reward.mean().item()                             # [LOG]
            epoch_d_real += torch.sigmoid(real_logits).mean().item()        # [LOG]
            epoch_d_fake += torch.sigmoid(fake_logits).mean().item()        # [LOG]
            n_batches    += 1                                                 # [LOG]

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch}/{EPOCHS} | G Loss: {g_loss.item():.4f} | D Loss: {loss_d.item():.4f}")

        avg_g      = epoch_g_loss / max(n_batches, 1)                        # [LOG]
        avg_d      = epoch_d_loss / max(n_batches, 1)                        # [LOG]
        avg_reward = epoch_reward / max(n_batches, 1)                        # [LOG]
        avg_d_real = epoch_d_real / max(n_batches, 1)                        # [LOG]
        avg_d_fake = epoch_d_fake / max(n_batches, 1)                        # [LOG]
        log_writer.writerow(["adversarial", epoch, f"{avg_g:.6f}", f"{avg_d:.6f}", f"{avg_reward:.4f}", f"{avg_d_real:.4f}", f"{avg_d_fake:.4f}"])  # [LOG]
        log_file.flush()                                                      # [LOG]

    log_file.close()                                                          # [LOG]

    print("Salvataggio dei modelli in corso...")
    torch.save(generator.state_dict(), GENERATOR_MODEL)
    torch.save(discriminator.state_dict(), DISCRIMINATOR_MODEL)
    print("Addestramento completato e modelli salvati!")


if __name__ == "__main__":
    train_cgan()