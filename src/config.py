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
TOKENIZED_OUTPUT = str(DATA_DIR / "tokenized" / "error_based_tokenized.csv")
GENERATOR_MODEL = str(MODELS_DIR / "generator_model.pth")
DISCRIMINATOR_MODEL = str(MODELS_DIR / "discriminator_model.pth")

# Crea le directory
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ==================== MODEL ====================
VOCAB_SIZE = 379
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 3

# Token speciali
PAD_TOKEN = 0
START_TOKEN = 2


# ==================== TRAINING ====================
BATCH_SIZE = 32
MAX_SEQ_LEN = 50
EPOCHS = 50
PRETRAIN_EPOCHS = 100

LEARNING_RATE_G = 0.0005  # Ridotto per maggiore stabilità con la capacità del modello aumentata
LEARNING_RATE_D = 0.0005

GRAD_CLIP = 5.0

USE_CUDA = True


# ==================== DATA ====================
SHUFFLE = True
DROP_LAST_BATCH = True

# ==================== GENERATION ====================
NUM_SAMPLES = 10
GENERATION_LABEL_TYPE = 0  # 0=error_based, 1=time_based, 2=union_based
GENERATION_TEMPERATURE = 0.8  # Temperatura di campionamento (più basso = più deterministico)

# ==================== BPE TOKENIZER ====================
BPE_VOCAB_SIZE = 5000
BPE_MIN_FREQUENCY = 2
BPE_OUTPUT = str(DATA_DIR / "bpe_config" / "sql_bpe_tokenizer_config.json")
