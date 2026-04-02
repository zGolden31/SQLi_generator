import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GENERATION_TEMPERATURE


class ConditionalGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(ConditionalGenerator, self).__init__()
        self.hidden_dim = hidden_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        # LSTM input: token embedding concatenated with label embedding
        self.lstm = nn.LSTM(embed_dim * 2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, labels, hidden=None):
        batch_size, seq_len = x.size()

        emb_x = self.token_embedding(x)  # (B, T, embed_dim)

        emb_label = self.label_embedding(labels)  # (B, 1, embed_dim) or (B, embed_dim)
        if emb_label.dim() == 2:
            emb_label = emb_label.unsqueeze(1)
        emb_label = emb_label.expand(batch_size, seq_len, -1)  # (B, T, embed_dim)

        lstm_input = torch.cat([emb_x, emb_label], dim=2)  # (B, T, embed_dim * 2)
        out, hidden = self.lstm(lstm_input, hidden)          # (B, T, hidden_dim)
        logits = self.fc(out)                                 # (B, T, vocab_size)

        return logits, hidden

    def sample(self, batch_size, start_token_id, labels, max_seq_len):
        """Autoregressively generate a sequence of tokens starting from start_token_id."""
        inputs = torch.LongTensor(batch_size, 1).fill_(start_token_id).to(labels.device)

        if next(self.parameters()).is_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        hidden = None
        samples = []

        for _ in range(max_seq_len):
            logits, hidden = self.forward(inputs, labels, hidden)
            logits = logits[:, -1, :] / GENERATION_TEMPERATURE  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)             # (B, 1)
            samples.append(next_token)
            inputs = next_token

        return torch.cat(samples, dim=1)  # (B, max_seq_len)    