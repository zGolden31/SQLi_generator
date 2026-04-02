import torch
from tokenizers import Tokenizer
from generator import ConditionalGenerator
from config import (
    VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES,
    GENERATOR_MODEL, TOKENIZER_CONFIG,
    START_TOKEN, MAX_SEQ_LEN,
    NUM_SAMPLES, GENERATION_LABEL_TYPE
)


def generate_payloads(num_samples=None, label_type=None, max_seq_len=None):
    num_samples = NUM_SAMPLES if num_samples is None else num_samples
    label_type = GENERATION_LABEL_TYPE if label_type is None else label_type
    max_seq_len = MAX_SEQ_LEN if max_seq_len is None else max_seq_len

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(TOKENIZER_CONFIG)

    print("Loading generator...")
    generator = ConditionalGenerator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
    generator.load_state_dict(torch.load(GENERATOR_MODEL, map_location=device))
    generator.to(device)
    generator.eval()

    print(f"\nGenerating {num_samples} payloads for class {label_type}...\n")
    labels = torch.tensor([label_type] * num_samples).to(device)

    with torch.no_grad():
        generated_tokens = generator.sample(
            batch_size=num_samples,
            start_token_id=START_TOKEN,
            labels=labels,
            max_seq_len=max_seq_len
        )

    for i, token_sequence in enumerate(generated_tokens.cpu().tolist()):
        decoded = tokenizer.decode(token_sequence, skip_special_tokens=True)
        print(f"Payload {i+1} ---")
        print(decoded.replace(" ", ""))
        print("-" * 30)


if __name__ == "__main__":
    generate_payloads(num_samples=10, label_type=0)