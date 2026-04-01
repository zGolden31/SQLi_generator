import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from config import CSV_FILE, BPE_VOCAB_SIZE, BPE_MIN_FREQUENCY, BPE_OUTPUT

def train_bpe_tokenizer(data_path, output_path="data/bpe_config/sql_bpe_tokenizer_config.json"):
    # Usiamo un token [UNK] per gestire i token non visti durante l'addestramento
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Normalizzazione: rende il testo più uniforme (es. minuscole, rimozione spazi extra)
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Lowercase()
    ])

    # Pre-tokenizzazione: 
    # Utilizziamo lo stesso pattern regex utilizzato in tokenizator.py 
    sql_pattern = r'(\[[A-Z0-9_]+\])|(0x[0-9A-Fa-f]+)|(\d+)|([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)|([()=/*+,;-])'

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(), # Divide il testo in base agli spazi
         pre_tokenizers.Split(sql_pattern, behavior="isolated") # Isola i componenti SQL definiti
    ])

    # Configurazione del trainer
    # Impostiamo la dimension del vocabolario e i token speciali
    trainer = trainers.BpeTrainer(
        vocab_size=BPE_VOCAB_SIZE, # Dimensione del vocabolario
        min_frequency=BPE_MIN_FREQUENCY, # Frequenza minima per includere un token nel vocabolario
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] # Token speciali
    )

    # Caricamente dei dati per il training
    # Carichiamo il dataset CSV 
    df = pd.read_csv(data_path)
    if 'request/payload' not in df.columns:
        raise ValueError("Il dataset deve contenere una colonna 'request/payload'")
    payloads = df['request/payload'].astype(str).tolist()

    # Training
    tokenizer.train_from_iterator(payloads, trainer=trainer)

    # Salvataggio del tokenizer addestrato
    tokenizer.save(output_path)
    print(f"Tokenizer BPE addestrato e salvato in {output_path}")

    return tokenizer

# Esempio di utilizzo
data_file = CSV_FILE
train_bpe_tokenizer(data_file, BPE_OUTPUT)


