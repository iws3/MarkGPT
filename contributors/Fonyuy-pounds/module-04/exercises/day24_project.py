"""
Day 24 Project: Mini-Project 4 - Character-Level LSTM with Attention
Module 04: Sequence Modeling
=====================================================================

Objective:
- Build a character-level LSTM with Attention trained on the Gospel of Mark.
- Generate 200-character passages that continue a given seed verse.
- Compare quality with previous character-level models (like Day 20 RNN).

Background:
This is your baseline model before we move to Transformers in Module 06.
By combining the memory of an LSTM with the selective focus of Attention,
we should see significantly more coherent text than the vanilla RNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import json
import numpy as np

# --- Hyperparameters ---
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 128
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
SEQ_LENGTH = 100  # Length of the input seed/verse
GENERATE_LENGTH = 200 # Length of the generated passage
TEACHER_FORCING_RATIO = 0.5

# --- Data Loading ---
def load_data():
    workspace_root = r'c:\Users\the eye informatique\Desktop\ML\AI\MarkGPT'
    data_dir = os.path.join(workspace_root, 'data', 'processed', 'mark_char')
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found. Ensure preprocess_char.py was run on mark.txt.")

    with open(os.path.join(data_dir, 'meta.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    train_data = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16)
    val_data = np.fromfile(os.path.join(data_dir, 'val.bin'), dtype=np.uint16)
    
    return train_data, val_data, meta

# --- Model Architecture ---

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        # LSTM returns (outputs, (hidden, cell))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # Using dot-product attention
        
    def forward(self, hidden, encoder_outputs):
        # hidden (from LSTM) is (num_layers, batch, hidden_size)
        # We take the top layer: (batch, hidden_size)
        query = hidden[-1].unsqueeze(2) # (batch, hidden_size, 1)
        
        # encoder_outputs: (batch, seq_len, hidden_size)
        # Dot product: (batch, seq_len, hidden_size) @ (batch, hidden_size, 1) -> (batch, seq_len, 1)
        energy = torch.bmm(encoder_outputs, query).squeeze(2) # (batch, seq_len)
        
        weights = F.softmax(energy, dim=1)
        return weights

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(embedding_size + hidden_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden_state, encoder_outputs):
        # x: (batch, 1)
        embedded = self.embedding(x)
        
        # Calculate attention
        hidden, cell = hidden_state
        weights = self.attention(hidden, encoder_outputs).unsqueeze(1) # (batch, 1, seq_len)
        
        # Context vector: (batch, 1, seq_len) @ (batch, seq_len, hidden_size) -> (batch, 1, hidden_size)
        context = torch.bmm(weights, encoder_outputs)
        
        # Combine input and context
        lstm_input = torch.cat((embedded, context), dim=2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Concatenate output and context for final prediction
        prediction = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))
        
        return prediction, (hidden, cell), weights.squeeze(1)

class Seq2SeqMark(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqMark, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target_len, teacher_forcing_ratio=0.5, target=None):
        batch_size = source.shape[0]
        vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(source.device)
        
        encoder_outputs, hidden_state = self.encoder(source)
        
        # Start token (we'll use a space or a specific character as start)
        # For simplicity, we seed with the last character of the source
        x = source[:, -1].unsqueeze(1)
        
        for t in range(target_len):
            prediction, hidden_state, _ = self.decoder(x, hidden_state, encoder_outputs)
            outputs[:, t, :] = prediction
            
            # Teacher forcing
            if target is not None and random.random() < teacher_forcing_ratio:
                x = target[:, t].unsqueeze(1)
            else:
                x = prediction.argmax(1).unsqueeze(1)
                
        return outputs

# --- Training Loop ---

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...")
    
    train_data, val_data, meta = load_data()
    vocab_size = meta['vocab_size']
    stoi = meta['stoi']
    itos = {int(k): v for k, v in meta['itos'].items()}
    
    encoder = Encoder(vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE).to(device)
    decoder = AttentionDecoder(vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE).to(device)
    model = Seq2SeqMark(encoder, decoder).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        
        # Randomly sample batches
        for _ in range(100): # 100 batches per epoch
            idx = np.random.randint(0, len(train_data) - SEQ_LENGTH - GENERATE_LENGTH - 1, BATCH_SIZE)
            
            src_batch = []
            trg_batch = []
            for i in idx:
                src_batch.append(train_data[i : i + SEQ_LENGTH])
                trg_batch.append(train_data[i + SEQ_LENGTH : i + SEQ_LENGTH + GENERATE_LENGTH])
            
            src = torch.tensor(src_batch).to(device)
            trg = torch.tensor(trg_batch).to(device)
            
            optimizer.zero_grad()
            output = model(src, GENERATE_LENGTH, TEACHER_FORCING_RATIO, trg)
            
            # Flatten for loss
            output = output.view(-1, vocab_size)
            trg = trg.view(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch}/{NUM_EPOCHS}, Loss: {total_loss/100:.4f}")
        
        if epoch % 5 == 0:
            generate_sample(model, train_data, itos, device)

def generate_sample(model, data, itos, device):
    model.eval()
    with torch.no_grad():
        # Pick a random verse-start from data
        idx = np.random.randint(0, len(data) - SEQ_LENGTH)
        seed = data[idx : idx + SEQ_LENGTH]
        src = torch.tensor(seed).unsqueeze(0).to(device)
        
        output = model(src, GENERATE_LENGTH, teacher_forcing_ratio=0)
        output_ids = output.argmax(2).squeeze(0).cpu().numpy()
        
        seed_text = "".join([itos[i] for i in seed])
        generated_text = "".join([itos[i] for i in output_ids])
        
        print("\n--- SAMPLE GENERATION ---")
        print(f"Seed: {seed_text}")
        print(f"Continuation: {generated_text}")
        print("--------------------------\n")

if __name__ == "__main__":
    train()
