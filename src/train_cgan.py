import csv                        # Libreria standard per scrivere file CSV
import torch                      # Framework deep learning principale
import torch.nn as nn             # Moduli per reti neurali (loss functions, layers)
import torch.optim as optim       # Ottimizzatori (Adam, SGD, ...)
from pathlib import Path          # Gestione path cross-platform (Windows/Linux/Mac)

from config import (
    VOCAB_SIZE,           # Numero totale di token nel vocabolario BPE
    EMBED_DIM,            # Dimensione dei vettori di embedding (token e label)
    HIDDEN_DIM,           # Dimensione dello stato nascosto della LSTM
    NUM_CLASSES,          # Numero di classi condizionali (es. 3: error/time/union)
    GENERATOR_MODEL,      # Path dove salvare i pesi del generatore (.pth)
    DISCRIMINATOR_MODEL,  # Path dove salvare i pesi del discriminatore (.pth)
    TOKENIZER_CONFIG,     # Path del file JSON con la configurazione del tokenizer BPE
    CSV_FILE,             # Path del CSV con i payload reali di sqlmap
    BATCH_SIZE,           # Numero di sequenze processate per ogni step di training
    MAX_SEQ_LEN,          # Lunghezza massima (in token) di ogni sequenza
    EPOCHS,               # Numero di epoche della fase avversaria
    PRETRAIN_EPOCHS,      # Numero di epoche per il pre-training di G e D
    LEARNING_RATE_G,      # Learning rate dell'ottimizzatore del generatore
    LEARNING_RATE_D,      # Learning rate dell'ottimizzatore del discriminatore
    START_TOKEN,          # ID del token [CLS], usato come primo input della generazione
    GRAD_CLIP,            # Soglia massima per il gradient clipping (evita exploding gradients)
    MODELS_DIR            # Directory dove salvare modelli e log
)

from generator import ConditionalGenerator        # Generatore LSTM condizionale
from discriminator import ConditionalDiscriminator # Discriminatore BiLSTM condizionale
from dataset_loader import get_dataloader          # Funzione che costruisce il DataLoader


def train_cgan():

    # -----------------------------------------------------------------------
    # INIZIALIZZAZIONE
    # -----------------------------------------------------------------------

    # Istanzia il generatore: dato un label (tecnica SQLi) e un token iniziale,
    # genera autoregressivamente una sequenza di token SQL
    generator = ConditionalGenerator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)

    # Istanzia il discriminatore: data una sequenza di token e un label,
    # restituisce un logit che indica quanto la sequenza sembra reale
    discriminator = ConditionalDiscriminator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)

    # Usa la GPU se disponibile, altrimenti CPU
    # La GPU accelera significativamente il training su dataset grandi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # Adam: ottimizzatore con learning rate adattivo per parametro
    # Usato per entrambi i modelli con learning rate separati
    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D)

    # CrossEntropyLoss per il pre-training MLE del generatore:
    # misura quanto le distribuzioni di probabilità predette si discostano
    # dai token reali. ignore_index=0 esclude i token di padding dal calcolo
    g_pretrain_loss = nn.CrossEntropyLoss(ignore_index=0)

    # BCEWithLogitsLoss per il discriminatore: combina sigmoid + binary cross entropy
    # in un'unica operazione numericamente stabile. Usata per classificare
    # sequenze come reali (1) o generate (0)
    d_loss_fn = nn.BCEWithLogitsLoss()

    print("Caricamento dataset in corso...")

    # Costruisce il DataLoader: legge il CSV, tokenizza i payload con BPE,
    # applica padding/truncation a MAX_SEQ_LEN e restituisce batch di tensori
    # label_value=0 indica che tutti i payload di questo file sono 'error_based'
    dataloader = get_dataloader(
        csv_file=CSV_FILE,
        tokenizer_config_path=TOKENIZER_CONFIG,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        label_value=0
    )

    # -----------------------------------------------------------------------
    # LOG: apertura file CSV di training
    # Un unico file per tutto il training, con una riga per epoca per fase.
    # Le colonne vuote ("") indicano metriche non applicabili a quella fase.
    # -----------------------------------------------------------------------
    log_path = Path(MODELS_DIR) / "training_log.csv"          # Path completo del file di log
    log_file = open(log_path, "w", newline="", encoding="utf-8") # Apre in scrittura, sovrascrive se esiste
    log_writer = csv.writer(log_file)                            # Writer CSV che gestisce virgole e quoting
    log_writer.writerow([                                        # Scrive la riga di intestazione
        "phase",        # Fase corrente: pretrain_g | pretrain_d | adversarial
        "epoch",        # Numero dell'epoca (1-based per pretrain, 0-based per adversarial)
        "g_loss",       # Loss media del generatore nell'epoca
        "d_loss",       # Loss media del discriminatore nell'epoca
        "reward_mean",  # Reward medio assegnato dal discriminatore ai payload generati
        "d_real_mean",  # P(reale | payload reale): vicino a 1 = D riconosce i reali
        "d_fake_mean"   # P(reale | payload generato): vicino a 0 = D riconosce i falsi
    ])

    # -----------------------------------------------------------------------
    # FASE 1 — Pre-training del Generatore (MLE, Maximum Likelihood Estimation)
    #
    # Il generatore impara a riprodurre i payload reali token per token.
    # Tecnica: teacher forcing — l'input è la sequenza reale shiftata di 1,
    # il target è la sequenza reale originale. Il modello impara a predire
    # il token successivo dato il contesto precedente reale (non generato).
    # Questo fornisce una base linguistica prima dell'addestramento avversario.
    # -----------------------------------------------------------------------
    print("Inizio del pre-training del generatore...")
    for pre_epoch in range(PRETRAIN_EPOCHS):
        epoch_g_loss = 0.0  # Accumula la loss di tutti i batch dell'epoca
        n_batches = 0       # Conta i batch per calcolare la media a fine epoca

        for real_data, labels in dataloader:
            real_data, labels = real_data.to(device), labels.to(device)  # Sposta i tensori sul device

            g_optimizer.zero_grad()  # Azzera i gradienti accumulati dallo step precedente

            # Teacher forcing: l'input è token[0..T-2], il target è token[1..T-1]
            # In questo modo ogni posizione t predice il token t+1
            inputs  = real_data[:, :-1]  # Tutti i token tranne l'ultimo: shape (B, T-1)
            targets = real_data[:, 1:]   # Tutti i token tranne il primo:  shape (B, T-1)

            # Forward pass: logits ha shape (B, T-1, vocab_size)
            logits, _ = generator(inputs, labels)

            # Reshape per CrossEntropyLoss: richiede (N, C) e (N,)
            # dove N = B*(T-1) e C = vocab_size
            loss = g_pretrain_loss(
                logits.reshape(-1, VOCAB_SIZE),  # (B*(T-1), vocab_size)
                targets.reshape(-1)              # (B*(T-1),)
            )

            loss.backward()  # Calcola i gradienti con backpropagation

            # Gradient clipping: limita la norma del gradiente a GRAD_CLIP
            # Previene l'exploding gradient problem tipico delle LSTM
            nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP)

            g_optimizer.step()  # Aggiorna i pesi con Adam

            epoch_g_loss += loss.item()  # Accumula la loss scalare (detached dal grafo)
            n_batches += 1               # Incrementa il contatore di batch

        avg_g = epoch_g_loss / max(n_batches, 1)  # Loss media dell'epoca (max evita divisione per zero)

        # Scrive una riga nel CSV: d_loss, reward e metriche D vuote (non applicabili)
        log_writer.writerow(["pretrain_g", pre_epoch + 1, f"{avg_g:.6f}", "", "", "", ""])
        log_file.flush()  # Forza la scrittura su disco: se il training crasha, i dati sono salvati

        print(f"Pre-training G - Epoca {pre_epoch+1}/{PRETRAIN_EPOCHS} completata.")

    # -----------------------------------------------------------------------
    # FASE 2 — Pre-training del Discriminatore
    #
    # Il discriminatore impara a distinguere payload reali da quelli generati
    # dal generatore già pre-trainato. Usare G pre-trainato (non rumore puro)
    # rende il problema più realistico fin da subito.
    # Target: 1 per payload reali, 0 per payload generati.
    # -----------------------------------------------------------------------
    print("Inizio del pre-training del discriminatore...")
    for pre_epoch in range(PRETRAIN_EPOCHS):
        epoch_d_loss = 0.0  # Accumula la loss totale (reale + falso) del discriminatore
        epoch_d_real = 0.0  # Accumula P(reale | payload reale): salute del discriminatore
        epoch_d_fake = 0.0  # Accumula P(reale | payload generato): salute del discriminatore
        n_batches = 0

        for real_data, labels in dataloader:
            real_data, labels = real_data.to(device), labels.to(device)

            d_optimizer.zero_grad()

            # --- Campione reale: il discriminatore deve predire 1 ---
            real_logits  = discriminator(real_data, labels)           # Logit per i payload reali
            real_targets = torch.ones_like(real_logits)               # Target = 1 (reale)
            loss_real    = d_loss_fn(real_logits, real_targets)       # BCE loss sul campione reale

            # --- Campione generato: il discriminatore deve predire 0 ---
            # Il generatore genera senza calcolare gradienti (torch.no_grad):
            # in questa fase vogliamo aggiornare solo il discriminatore
            fake_data    = generator.sample(
                batch_size=BATCH_SIZE,
                start_token_id=START_TOKEN,
                labels=labels,
                max_seq_len=MAX_SEQ_LEN
            )
            fake_logits  = discriminator(fake_data.detach(), labels)  # detach(): blocca gradienti verso G
            fake_targets = torch.zeros_like(fake_logits)              # Target = 0 (falso)
            loss_fake    = d_loss_fn(fake_logits, fake_targets)       # BCE loss sul campione generato

            # Loss totale del discriminatore: somma delle due BCE
            loss_d = loss_real + loss_fake
            loss_d.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP)
            d_optimizer.step()

            epoch_d_loss += loss_d.item()                              # Accumula loss totale
            epoch_d_real += torch.sigmoid(real_logits).mean().item()  # P(reale|reale): target ~0.7
            epoch_d_fake += torch.sigmoid(fake_logits).mean().item()  # P(reale|fake): target ~0.3
            n_batches += 1

        avg_d      = epoch_d_loss / max(n_batches, 1)
        avg_d_real = epoch_d_real / max(n_batches, 1)
        avg_d_fake = epoch_d_fake / max(n_batches, 1)

        # Scrive una riga nel CSV: g_loss e reward vuoti (non applicabili in questa fase)
        log_writer.writerow(["pretrain_d", pre_epoch + 1, "", f"{avg_d:.6f}", "", f"{avg_d_real:.4f}", f"{avg_d_fake:.4f}"])
        log_file.flush()

        print(f"Pre-training D - Epoca {pre_epoch+1}/{PRETRAIN_EPOCHS} completata.")

    # -----------------------------------------------------------------------
    # FASE 3 — Training Avversario con Policy Gradient (REINFORCE)
    #
    # Il generatore (policy) campiona sequenze di token (azioni).
    # Il discriminatore assegna un reward a ogni sequenza.
    # Il generatore viene aggiornato per massimizzare il reward atteso:
    #   L_G = -E[advantage * log P(token)]
    # dove advantage = reward - baseline riduce la varianza del gradiente.
    #
    # Il discriminatore viene aggiornato in parallelo per non perdere
    # capacità discriminativa man mano che G migliora.
    # -----------------------------------------------------------------------
    print("Inizio Addestramento Avversario (Reinforcement Learning)...")

    # Baseline EMA (Exponential Moving Average) del reward:
    # inizia a 0.5 (centro del range sigmoid) e si aggiorna ad ogni batch.
    # Sottraendo la baseline al reward si ottiene l'advantage, che ha media ~0
    # e varianza ridotta → gradienti più stabili per il generatore
    reward_baseline = 0.5

    for epoch in range(EPOCHS):
        epoch_g_loss = 0.0  # Accumula la policy gradient loss del generatore
        epoch_d_loss = 0.0  # Accumula la BCE loss del discriminatore
        epoch_reward = 0.0  # Accumula il reward medio (sigmoid output del D)
        epoch_d_real = 0.0  # Accumula P(reale | payload reale)
        epoch_d_fake = 0.0  # Accumula P(reale | payload generato)
        n_batches = 0

        for batch_idx, (real_data, labels) in enumerate(dataloader):
            real_data, labels = real_data.to(device), labels.to(device)

            # ---------------------------------------------------------------
            # A. UPDATE DEL GENERATORE — Policy Gradient (REINFORCE)
            # ---------------------------------------------------------------
            g_optimizer.zero_grad()

            # Il generatore campiona una sequenza completa autoregressivamente:
            # parte da START_TOKEN e genera MAX_SEQ_LEN token uno alla volta
            fake_data = generator.sample(
                batch_size=BATCH_SIZE,
                start_token_id=START_TOKEN,
                labels=labels,
                max_seq_len=MAX_SEQ_LEN
            )

            # Il discriminatore valuta i payload generati e assegna un reward:
            # sigmoid(logit) ∈ (0,1): vicino a 1 = il D ritiene il payload reale
            # detach(): il reward è una costante per il PG, i gradienti non fluiscono nel D
            reward = torch.sigmoid(discriminator(fake_data, labels)).detach()

            # Aggiorna la baseline con EMA: decay 0.95 su storia, 0.05 sul valore corrente
            # La baseline segue lentamente il reward medio, riducendone la varianza
            reward_baseline = 0.95 * reward_baseline + 0.05 * reward.mean().item()

            # Advantage: reward centrato sulla baseline
            # Valori positivi → il batch è sopra la media → rinforzare quei token
            # Valori negativi → il batch è sotto la media → penalizzare quei token
            advantage = reward - reward_baseline

            # Riallineamento input/target per il calcolo delle log-probabilità:
            # pg_inputs: [START, token_0, ..., token_{T-2}] → input al generatore
            # fake_data: [token_0, token_1, ..., token_{T-1}] → token da valutare
            # In questo modo pg_inputs[t] genera fake_data[t] per ogni t
            start_tokens = torch.LongTensor(BATCH_SIZE, 1).fill_(START_TOKEN).to(device)
            pg_inputs = torch.cat([start_tokens, fake_data[:, :-1]], dim=1)  # (B, T)

            # Forward pass sul generatore per ottenere i logit di ogni token
            logits, _ = generator(pg_inputs, labels)  # (B, T, vocab_size)

            # Converti i logit in log-probabilità normalizzate sul vocabolario
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (B, T, vocab_size)

            # Estrai la log-probabilità del token effettivamente campionato a ogni posizione
            # gather seleziona, per ogni (batch, posizione), il valore all'indice fake_data[b,t]
            chosen_log_probs = log_probs.gather(
                dim=2,
                index=fake_data.unsqueeze(2)   # (B, T, 1) → indice del token scelto
            ).squeeze(2)                        # (B, T) → log P(token_t | contesto)

            # Loss REINFORCE: -E[advantage * log P(token)]
            # advantage.unsqueeze(1): broadcast da (B,) a (B,T) — stesso advantage per tutti i token
            # Il segno negativo converte da gradient ascent (massimizza reward) a gradient descent
            g_loss = -(advantage.unsqueeze(1) * chosen_log_probs).mean()

            g_loss.backward()   # Calcola i gradienti della policy gradient loss
            g_optimizer.step()  # Aggiorna i pesi del generatore

            # ---------------------------------------------------------------
            # B. UPDATE DEL DISCRIMINATORE — BCE su reale vs generato
            # Il discriminatore viene aggiornato ad ogni batch per mantenere
            # la sua capacità discriminativa man mano che G migliora
            # ---------------------------------------------------------------
            d_optimizer.zero_grad()

            # Valuta i payload reali: il D deve assegnare logit alto (→ 1)
            real_logits = discriminator(real_data, labels)

            # Valuta i payload appena generati: il D deve assegnare logit basso (→ 0)
            # detach(): i gradienti non devono fluire nel generatore durante l'update del D
            fake_logits = discriminator(fake_data.detach(), labels)

            # Loss totale del D: somma delle BCE su reali e generati
            loss_d = d_loss_fn(real_logits, torch.ones_like(real_logits)) + \
                     d_loss_fn(fake_logits, torch.zeros_like(fake_logits))

            loss_d.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP)
            d_optimizer.step()

            # Accumulo metriche di batch per la media di epoca
            epoch_g_loss += g_loss.item()                              # Policy gradient loss
            epoch_d_loss += loss_d.item()                              # BCE loss totale del D
            epoch_reward += reward.mean().item()                       # Reward medio grezzo (pre-baseline)
            epoch_d_real += torch.sigmoid(real_logits).mean().item()  # P(reale|reale): sano se ~0.7
            epoch_d_fake += torch.sigmoid(fake_logits).mean().item()  # P(reale|fake): sano se ~0.3
            n_batches += 1

        # Stampa a terminale ogni 5 epoche e all'ultima per non intasare l'output
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch}/{EPOCHS} | G Loss: {g_loss.item():.4f} | D Loss: {loss_d.item():.4f}")

        # Calcola medie di epoca dividendo per il numero di batch processati
        avg_g      = epoch_g_loss / max(n_batches, 1)
        avg_d      = epoch_d_loss / max(n_batches, 1)
        avg_reward = epoch_reward / max(n_batches, 1)
        avg_d_real = epoch_d_real / max(n_batches, 1)
        avg_d_fake = epoch_d_fake / max(n_batches, 1)

        # Scrive tutte le metriche dell'epoca avversaria nel CSV
        log_writer.writerow(["adversarial", epoch, f"{avg_g:.6f}", f"{avg_d:.6f}", f"{avg_reward:.4f}", f"{avg_d_real:.4f}", f"{avg_d_fake:.4f}"])
        log_file.flush()  # Scrittura immediata su disco

    log_file.close()  # Chiude il file CSV al termine del training

    # -----------------------------------------------------------------------
    # SALVATAGGIO DEI MODELLI
    # Salva solo i pesi (state_dict), non l'intera architettura:
    # più leggero e compatibile con future modifiche ai modelli
    # -----------------------------------------------------------------------
    print("Salvataggio dei modelli in corso...")
    torch.save(generator.state_dict(), GENERATOR_MODEL)
    torch.save(discriminator.state_dict(), DISCRIMINATOR_MODEL)
    print("Addestramento completato e modelli salvati!")


if __name__ == "__main__":
    train_cgan()