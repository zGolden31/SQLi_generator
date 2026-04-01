import torch
from tokenizers import Tokenizer
from generator import ConditionalGenerator
from config import (
    VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES,
    GENERATOR_MODEL, TOKENIZER_CONFIG, 
    START_TOKEN, USE_CUDA, MAX_SEQ_LEN,
    NUM_SAMPLES, GENERATION_LABEL_TYPE
)

def generate_payloads(num_samples=None, label_type=None, max_seq_len=None):
    # Configurazione dei parametri del modello
    num_samples = NUM_SAMPLES if num_samples is None else num_samples
    label_type = GENERATION_LABEL_TYPE if label_type is None else label_type
    max_seq_len = MAX_SEQ_LEN if max_seq_len is None else max_seq_len

    # Percorsi dei file
    model_path = GENERATOR_MODEL
    tokenizer_json = TOKENIZER_CONFIG

    # Usa la GPU se disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Caricamento del tokenizer
    print("Caricamento del tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_json)

    # Caricamento del modello
    print("Caricamento del Generatore...")
    generator = ConditionalGenerator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)

    # Carichiamo i "pesi" salvati durante l'addestramento
    # map_location=device serve per evitare errori se hai addestrato su GPU ma esegui su CPU
    generator.load_state_dict(torch.load(GENERATOR_MODEL, map_location=device))
    generator.to(device)

    # Impostiamo il modello in modalità "Valutazione/Inferenza" (disabilita il calcolo dei gradienti)
    generator.eval()

    # Generezione dei payloads
    print(f"\nGenerazione di {num_samples} payload per la classe {label_type}...\n")

    # Creiamo le etichette per i campioni che vogliamo generare (es. 0 per error_based)
    labels = torch.tensor([label_type] * num_samples).to(device)

    # Usiamo torch.no_grad() perché non dobbiamo aggiornare i pesi della rete ora
    with torch.no_grad():
        # L'ID 2 corrisponde a [CLS] nel vocabolario BPE
       generated_tokens = generator.sample(
            batch_size=num_samples, 
            start_token_id=START_TOKEN, 
            labels=labels, 
            max_seq_len=max_seq_len
        ) 
       
    # Decodifica e stampa
    generated_tokens_list = generated_tokens.cpu().tolist()

    for i, token_sequence in enumerate(generated_tokens_list): 
        # Il Tokenizer ha una funzione comoda 'decode' che ritraduce gli ID in stringhe.
        # skip_special_tokens=True rimuove in automatico [PAD], [CLS], ecc.
        testo_decodificato = tokenizer.decode(token_sequence, skip_special_tokens=True)

        print(f"Payload {i+1} ---")
        # Rimuoviamo gli spazi extra che il tokenizer potrebbe aver aggiunto durante la decodifica
        print(testo_decodificato.replace(" ", ""))
        print("-" * 30)


if __name__ == "__main__":
    generate_payloads(num_samples=10, label_type=0)