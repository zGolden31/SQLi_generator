from tokenizers import Tokenizer

# Carica il tokenizer appena creato
tokenizer = Tokenizer.from_file("data/bpe_config/sql_bpe_tokenizer_config.json")

# Un payload tipico per testare la segmentazione
payload = "AND 1=1 UNION SELECT 0x41414141, [RANDNUM]"

# Codifica
encoded = tokenizer.encode(payload)

print(f"Payload originale: {payload}")
print(f"Tokens generati: {encoded.tokens}")
print(f"ID numerici: {encoded.ids}")