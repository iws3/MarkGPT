"""
Day 23 Exercise: Attention Mechanisms
Module 04: Sequence Modeling
=====================================================

Task E23.1: Implement Attention in a PyTorch Seq2Seq Model.
- Build an Encoder that returns all hidden states, not just the final one.
- Implement a Dot-Product Attention mechanism.
- Build an Attention Decoder that uses the attention weights to focus on 
  specific encoder outputs at each time step.
- Task: Revisit Sequence Reversal (or simple translation) to observe how 
  attention weights align with input tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import string

# Hyperparameters
HIDDEN_SIZE = 128
EMBEDDING_SIZE = 64
BATCH_SIZE = 32
LEARNING_RATE = 0.005
NUM_EPOCHS = 1500 # Slightly fewer epochs needed with attention

# Simple character vocabulary
vocab = list(string.ascii_lowercase + ' ')
char_to_ix = {c: i for i, c in enumerate(vocab)}
ix_to_char = {i: c for i, c in enumerate(vocab)}
VOCAB_SIZE = len(vocab)
MAX_LEN = 10

def generate_dataset(num_samples):
    X = []
    Y = []
    for _ in range(num_samples):
        length = random.randint(3, MAX_LEN)
        s = ''.join(random.choice(vocab) for _ in range(length))
        s_padded = s.ljust(MAX_LEN, ' ')
        r_padded = s[::-1].ljust(MAX_LEN, ' ')
        
        x = [char_to_ix[c] for c in s_padded]
        y = [char_to_ix[c] for c in r_padded]
        X.append(x)
        Y.append(y)
    return torch.tensor(X), torch.tensor(Y)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        # return ALL outputs (seq_len x hidden_size), plus final hidden state
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # Using dot-product attention (no extra weights needed for pure dot product)
        # But we could also use a linear layer for Bahdanau attention
        
    def forward(self, hidden, encoder_outputs):
        # hidden shape: (1, batch_size, hidden_size) -> (batch_size, hidden_size, 1)
        # encoder_outputs shape: (batch_size, seq_len, hidden_size)
        
        hidden = hidden.squeeze(0).unsqueeze(2)
        
        # Calculate dot product: (batch_size, seq_len, hidden_size) @ (batch_size, hidden_size, 1)
        # -> (batch_size, seq_len, 1)
        energy = torch.bmm(encoder_outputs, hidden)
        energy = energy.squeeze(2) # (batch_size, seq_len)
        
        # Softmax gives attention weights (alpha)
        attention_weights = F.softmax(energy, dim=1)
        
        return attention_weights

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.attention = Attention(hidden_size)
        
        # GRU input is now embedding + context vector
        self.gru = nn.GRU(embedding_size + hidden_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden, encoder_outputs):
        # x: (batch_size, 1)
        embedded = self.embedding(x)
        
        # Calculate attention weights
        # hidden: (1, batch_size, hidden_size)
        a = self.attention(hidden, encoder_outputs) # (batch_size, seq_len)
        a = a.unsqueeze(1) # (batch_size, 1, seq_len)
        
        # Apply attention to encoder outputs to get context vector
        # (batch, 1, seq_len) @ (batch, seq_len, hidden_size) -> (batch, 1, hidden_size)
        context = torch.bmm(a, encoder_outputs)
        
        # Combine embedded input and context
        gru_input = torch.cat((embedded, context), dim=2)
        
        output, hidden = self.gru(gru_input, hidden)
        
        # Final output layer uses both output and context
        prediction = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))
        
        return prediction, hidden, a.squeeze(1)

class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, target_len, vocab_size)

        # Get all encoder outputs
        encoder_outputs, hidden = self.encoder(source)

        x = target[:, 0].unsqueeze(1)

        for t in range(target_len):
            output, hidden, _ = self.decoder(x, hidden, encoder_outputs)
            outputs[:, t, :] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            x = target[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs

def train():
    print("Initializing Seq2Seq with Attention...")
    X_train, Y_train = generate_dataset(5000)
    
    encoder = Encoder(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE)
    decoder = AttentionDecoder(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE)
    model = Seq2SeqAttention(encoder, decoder)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        indices = torch.randperm(X_train.shape[0])[:BATCH_SIZE]
        src = X_train[indices]
        trg = Y_train[indices]
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        output = output.view(-1, VOCAB_SIZE)
        trg = trg.view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 300 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")
            evaluate(model, "attention")

def evaluate(model, sentence):
    model.eval()
    with torch.no_grad():
        sentence = sentence[:MAX_LEN].ljust(MAX_LEN, ' ')
        indices = [char_to_ix[c] for c in sentence]
        src = torch.tensor(indices).unsqueeze(0)
        
        encoder_outputs, hidden = model.encoder(src)
        
        x = torch.tensor([[char_to_ix[' ']]])
        
        output_chars = []
        attention_history = []
        
        for _ in range(MAX_LEN):
            output, hidden, attn_weights = model.decoder(x, hidden, encoder_outputs)
            top1 = output.argmax(1)
            output_chars.append(ix_to_char[top1.item()])
            x = top1.unsqueeze(1)
            attention_history.append(attn_weights.squeeze().tolist())
            
        print(f"Input:  '{sentence}'")
        print(f"Output: '{''.join(output_chars)}'")
        # Note: attention_history contains the focus weights at each step, 
        # visualizing this would show a strong anti-diagonal correlation for reversal!
    model.train()

if __name__ == "__main__":
    train()
