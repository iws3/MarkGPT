"""
Day 22 Exercise: Sequence-to-Sequence (Seq2Seq) Models
Module 04: Sequence Modeling
=====================================================

Task E22.1: Build an Encoder-Decoder model in PyTorch.
- Transitioning to PyTorch because manual BPTT for Seq2Seq is overly complex.
- Implement an Encoder RNN that outputs a context vector.
- Implement a Decoder RNN that generates an output sequence from the context.
- Task: Sequence Reversal. The model learns to reverse character sequences 
  to prove it can compress information into the context vector and unpack it.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import string

# Hyperparameters
HIDDEN_SIZE = 128
EMBEDDING_SIZE = 64
BATCH_SIZE = 32
LEARNING_RATE = 0.005
NUM_EPOCHS = 2000

# Simple character vocabulary
vocab = list(string.ascii_lowercase + ' ')
char_to_ix = {c: i for i, c in enumerate(vocab)}
ix_to_char = {i: c for i, c in enumerate(vocab)}
VOCAB_SIZE = len(vocab)
MAX_LEN = 10

def generate_dataset(num_samples):
    """Generate random strings and their reversals for training."""
    X = []
    Y = []
    for _ in range(num_samples):
        # Generate random length string (3 to MAX_LEN)
        length = random.randint(3, MAX_LEN)
        s = ''.join(random.choice(vocab) for _ in range(length))
        
        # Pad strings to MAX_LEN
        s_padded = s.ljust(MAX_LEN, ' ')
        r_padded = s[::-1].ljust(MAX_LEN, ' ')
        
        # Convert to indices
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
        # We only care about the final hidden state (context vector)
        _, hidden = self.gru(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        # x shape: (batch_size, 1) - processing one step at a time
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.fc_out.out_features

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len, vocab_size)

        # Encoder context vector
        hidden = self.encoder(source)

        # First input to the decoder is a blank space (or a start token)
        x = target[:, 0].unsqueeze(1)

        for t in range(target_len):
            output, hidden = self.decoder(x, hidden)
            outputs[:, t, :] = output
            
            # Teacher forcing: decide whether to use actual next target or model prediction
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            x = target[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs

def train():
    print("Initializing Seq2Seq Sequence Reversal Training...")
    
    # Create dataset
    X_train, Y_train = generate_dataset(5000)
    
    # Initialize models
    encoder = Encoder(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE)
    decoder = Decoder(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE)
    model = Seq2Seq(encoder, decoder)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        # Sample a batch
        indices = torch.randperm(X_train.shape[0])[:BATCH_SIZE]
        src = X_train[indices]
        trg = Y_train[indices]
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        # Reshape for CrossEntropyLoss
        # output: (batch_size, target_len, vocab_size) -> (batch_size * target_len, vocab_size)
        output = output.view(-1, VOCAB_SIZE)
        trg = trg.view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")
            evaluate(model, "hello world")

def evaluate(model, sentence):
    model.eval()
    with torch.no_grad():
        sentence = sentence[:MAX_LEN].ljust(MAX_LEN, ' ')
        indices = [char_to_ix[c] for c in sentence]
        src = torch.tensor(indices).unsqueeze(0) # (1, seq_len)
        
        hidden = model.encoder(src)
        
        # Start token (using space arbitrarily as padding/start)
        x = torch.tensor([[char_to_ix[' ']]])
        
        output_chars = []
        for _ in range(MAX_LEN):
            output, hidden = model.decoder(x, hidden)
            top1 = output.argmax(1)
            output_chars.append(ix_to_char[top1.item()])
            x = top1.unsqueeze(1)
            
        print(f"Input:  '{sentence}'")
        print(f"Output: '{''.join(output_chars)}'")
    model.train()

if __name__ == "__main__":
    train()
