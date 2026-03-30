import pandas as pd
import re

def tokenize_sqli(payload):
    # Regole di tokenizzazione per SQLi
    token_pattern = re.compile(
        r'(\[[A-Z0-9_]+\])|' # 1. Segnaposti SQLMap: [RANDNUM], [ORIGVALUE]
        r'(0x[0-9A-Fa-f]+)|' # 2. Numeri esadecimali: 0x1A3F, 0x28, 0x3A
        r'(\d+)|' # 3. Numeri interi: 123, 45, 6789
        r'([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)|' # 4. Identificatori SQL: SELECT, FROM, users.name
        r'([()=/*+,;-])' # 5. Operatori e separatori: (), =, /*, +, -, ;
    )

    # Trova tutti i token ignorando gli spazi vuoti
    # La funzione finditer restituisce un iteratore di match, e per ogni match estraiamo il token con group(0)
    tokens = [match.group(0) for match in token_pattern.finditer(payload) if match.group(0).strip()]
    return tokens

def process_csv_dataset(file_path):
    # Carica il dataset CSV
    df = pd.read_csv(file_path)

    # Assicurati che ci sia una colonna 'payload'
    if 'request/payload' not in df.columns:
        raise ValueError("Il dataset deve contenere una colonna 'request/payload'")

    # Applica la tokenizzazione a ogni payload e crea una nuova colonna 'tokens'
    df['tokens'] = df['request/payload'].apply(tokenize_sqli)

    return df

dataset_tokenized = process_csv_dataset(r'data/raw/error_based.csv')
dataset_tokenized.to_csv(r'data/tokenized/error_based_tokenized.csv', index=False)
dataset_tokenized.to_json(r'data/tokenized/error_based_tokenized.json', orient='records', lines=True)
print(dataset_tokenized[['request/payload', 'tokens']].head())



