"""
Configurazione centralizzata per SQLi_generator
"""

import os
from pathlib import Path

# ==================== PATHS ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"

CSV_FILE = str(DATA_DIR / "raw" / "error_based.csv")
TOKENIZER_CONFIG = str(DATA_DIR / "bpe_config" / "sql_bpe_tokenizer_config.json")
GENERATOR_MODEL = str(MODELS_DIR / "generator_model.pth")
DISCRIMINATOR_MODEL = str(MODELS_DIR / "discriminator_model.pth")

# Crea le directory
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ==================== MODEL ====================
VOCAB_SIZE = 379
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_CLASSES = 3

# Token speciali
PAD_TOKEN = 0
START_TOKEN = 2


# ==================== TRAINING ====================
BATCH_SIZE = 32
MAX_SEQ_LEN = 50
EPOCHS = 50
PRETRAIN_EPOCHS = 5

LEARNING_RATE_G = 0.001
LEARNING_RATE_D = 0.001

USE_CUDA = True


# ==================== DATA ====================
SHUFFLE = True
DROP_LAST_BATCH = True