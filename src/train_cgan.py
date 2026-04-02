import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

from config import (
    VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES,
    GENERATOR_MODEL, DISCRIMINATOR_MODEL,
    TOKENIZER_CONFIG, CSV_FILE,
    BATCH_SIZE, MAX_SEQ_LEN, EPOCHS, PRETRAIN_EPOCHS,
    LEARNING_RATE_G, LEARNING_RATE_D,
    START_TOKEN, GRAD_CLIP, MODELS_DIR
)

from generator import ConditionalGenerator
from discriminator import ConditionalDiscriminator
from dataset_loader import get_dataloader


def train_cgan():

    generator = ConditionalGenerator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
    discriminator = ConditionalDiscriminator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D)

    g_pretrain_loss = nn.CrossEntropyLoss(ignore_index=0)
    d_loss_fn = nn.BCEWithLogitsLoss()

    print("Loading dataset...")
    dataloader = get_dataloader(
        csv_file=CSV_FILE,
        tokenizer_config_path=TOKENIZER_CONFIG,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        label_value=0
    )

    log_path = Path(MODELS_DIR) / "training_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["phase", "epoch", "g_loss", "d_loss", "reward_mean",
                         "d_real_mean", "d_fake_mean", "diversity"])

    # ------------------------------------------------------------------
    # Phase 1 — Generator pre-training (MLE with teacher forcing)
    # ------------------------------------------------------------------
    print("Starting generator pre-training...")
    for pre_epoch in range(PRETRAIN_EPOCHS):
        epoch_g_loss = 0.0
        n_batches = 0

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

            epoch_g_loss += loss.item()
            n_batches += 1

        avg_g = epoch_g_loss / max(n_batches, 1)
        log_writer.writerow(["pretrain_g", pre_epoch + 1, f"{avg_g:.6f}", "", "", "", "", ""])
        log_file.flush()
        print(f"Pre-training G - Epoch {pre_epoch+1}/{PRETRAIN_EPOCHS} done.")

    # ------------------------------------------------------------------
    # Phase 2 — Discriminator pre-training
    # ------------------------------------------------------------------
    print("Starting discriminator pre-training...")
    for pre_epoch in range(PRETRAIN_EPOCHS):
        epoch_d_loss = 0.0
        epoch_d_real = 0.0
        epoch_d_fake = 0.0
        n_batches = 0

        for real_data, labels in dataloader:
            real_data, labels = real_data.to(device), labels.to(device)
            d_optimizer.zero_grad()

            real_logits  = discriminator(real_data, labels)
            # Label smoothing: use 0.9 instead of 1.0 for real targets
            real_targets = torch.full_like(real_logits, 0.9)
            loss_real    = d_loss_fn(real_logits, real_targets)

            fake_data    = generator.sample(
                batch_size=BATCH_SIZE, start_token_id=START_TOKEN,
                labels=labels, max_seq_len=MAX_SEQ_LEN
            )
            fake_logits  = discriminator(fake_data.detach(), labels)
            fake_targets = torch.zeros_like(fake_logits)
            loss_fake    = d_loss_fn(fake_logits, fake_targets)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP)
            d_optimizer.step()

            epoch_d_loss += loss_d.item()
            epoch_d_real += torch.sigmoid(real_logits).mean().item()
            epoch_d_fake += torch.sigmoid(fake_logits).mean().item()
            n_batches += 1

        avg_d      = epoch_d_loss / max(n_batches, 1)
        avg_d_real = epoch_d_real / max(n_batches, 1)
        avg_d_fake = epoch_d_fake / max(n_batches, 1)
        log_writer.writerow(["pretrain_d", pre_epoch + 1, "", f"{avg_d:.6f}", "",
                              f"{avg_d_real:.4f}", f"{avg_d_fake:.4f}", ""])
        log_file.flush()
        print(f"Pre-training D - Epoch {pre_epoch+1}/{PRETRAIN_EPOCHS} done.")

    # ------------------------------------------------------------------
    # Phase 3 — Adversarial training (Policy Gradient / REINFORCE)
    #
    # The discriminator is updated every D_UPDATE_FREQ generator updates
    # to prevent it from overpowering the generator too quickly.
    # Label smoothing (0.9) is also applied to real targets.
    # ------------------------------------------------------------------
    print("Starting adversarial training (RL)...")

    # How often to update D relative to G (1 D update per D_UPDATE_FREQ G updates)
    D_UPDATE_FREQ = 3

    # Small entropy bonus to encourage diversity without dominating the loss
    ENTROPY_COEFF = 0.001

    reward_baseline = 0.5  # EMA baseline for advantage estimation

    for epoch in range(EPOCHS):
        epoch_g_loss  = 0.0
        epoch_d_loss  = 0.0
        epoch_reward  = 0.0
        epoch_d_real  = 0.0
        epoch_d_fake  = 0.0
        epoch_diversity = 0.0
        n_batches = 0

        for batch_idx, (real_data, labels) in enumerate(dataloader):
            real_data, labels = real_data.to(device), labels.to(device)

            # -- Generator update (Policy Gradient) --
            g_optimizer.zero_grad()

            fake_data = generator.sample(
                batch_size=BATCH_SIZE, start_token_id=START_TOKEN,
                labels=labels, max_seq_len=MAX_SEQ_LEN
            )

            reward = torch.sigmoid(discriminator(fake_data, labels)).detach()
            reward_baseline = 0.95 * reward_baseline + 0.05 * reward.mean().item()

            advantage = reward - reward_baseline
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            start_tokens = torch.LongTensor(BATCH_SIZE, 1).fill_(START_TOKEN).to(device)
            pg_inputs = torch.cat([start_tokens, fake_data[:, :-1]], dim=1)

            logits, _ = generator(pg_inputs, labels)
            log_probs = F.log_softmax(logits, dim=-1)
            probs     = F.softmax(logits, dim=-1)

            chosen_log_probs = log_probs.gather(
                dim=2, index=fake_data.unsqueeze(2)
            ).squeeze(2)

            pg_loss = -(advantage.unsqueeze(1) * chosen_log_probs).mean()
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            g_loss  = pg_loss - ENTROPY_COEFF * entropy

            g_loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP)
            g_optimizer.step()

            batch_diversity = sum(
                row.unique().numel() / row.numel() for row in fake_data
            ) / fake_data.size(0)

            # -- Discriminator update (every D_UPDATE_FREQ generator steps) --
            if batch_idx % D_UPDATE_FREQ == 0:
                d_optimizer.zero_grad()

                real_logits = discriminator(real_data, labels)
                fake_logits = discriminator(fake_data.detach(), labels)

                # Label smoothing on real targets to prevent overconfident discriminator
                loss_d = d_loss_fn(real_logits, torch.full_like(real_logits, 0.9)) + \
                         d_loss_fn(fake_logits, torch.zeros_like(fake_logits))

                loss_d.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP)
                d_optimizer.step()
                epoch_d_loss += loss_d.item()
            else:
                with torch.no_grad():
                    real_logits = discriminator(real_data, labels)
                    fake_logits = discriminator(fake_data.detach(), labels)
            epoch_g_loss    += g_loss.item()
            epoch_reward    += reward.mean().item()
            epoch_d_real    += torch.sigmoid(real_logits).mean().item()
            epoch_d_fake    += torch.sigmoid(fake_logits).mean().item()
            epoch_diversity += batch_diversity
            n_batches += 1

        avg_g         = epoch_g_loss    / max(n_batches, 1)
        avg_d         = epoch_d_loss    / max(n_batches, 1)
        avg_reward    = epoch_reward    / max(n_batches, 1)
        avg_d_real    = epoch_d_real    / max(n_batches, 1)
        avg_d_fake    = epoch_d_fake    / max(n_batches, 1)
        avg_diversity = epoch_diversity / max(n_batches, 1)

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch}/{EPOCHS} | G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f}")

        log_writer.writerow(["adversarial", epoch, f"{avg_g:.6f}", f"{avg_d:.6f}",
                              f"{avg_reward:.4f}", f"{avg_d_real:.4f}",
                              f"{avg_d_fake:.4f}", f"{avg_diversity:.4f}"])
        log_file.flush()

    log_file.close()

    print("Saving models...")
    torch.save(generator.state_dict(), GENERATOR_MODEL)
    torch.save(discriminator.state_dict(), DISCRIMINATOR_MODEL)
    print("Training complete and models saved!")


if __name__ == "__main__":
    train_cgan()