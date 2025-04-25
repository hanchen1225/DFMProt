import time
import torch
import numpy as np
from torch import nn, Tensor
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocabulary setup
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
idx_to_aa = {i: aa for aa, i in aa_to_idx.items()}
vocab_size = len(amino_acids)

pad_token = vocab_size                  
vocab_size += 1

max_seq_len = 1024

def load_fasta_sequences(path):
    sequences = []
    for record in SeqIO.parse(path, "fasta"):
        seq = str(record.seq)
        if len(seq) > 0:
            sequences.append(seq)
    return sequences

def pad_sequences(seqs, pad_token=pad_token):
    max_len = max(len(seq) for seq in seqs)
    padded = torch.full((len(seqs), max_len), pad_token, dtype=torch.long)
    lengths = torch.tensor([len(seq) for seq in seqs])
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    return padded, lengths

sequences = load_fasta_sequences("training_sequences.fasta")
print(f"Loaded {len(sequences)} sequences.")

lengths = [len(rec.seq) for rec in SeqIO.parse("training_sequences.fasta", "fasta")]
print(f"Total sequences: {len(lengths)}")
print(f"Max length: {max(lengths)}")
print(f"Min length: {min(lengths)}")
print(f"Mean length: {sum(lengths)/len(lengths):.1f}")
print(f"Median length: {sorted(lengths)[len(lengths)//2]}")

# Truncate during preprocessing
MAX_LEN = 1024
sequences = sequences[:MAX_LEN]

def tokenize_and_pad(seqs, max_len, pad_token):
    tokenized = []
    lengths = []
    for seq in seqs:
        token_ids = [aa_to_idx.get(aa, 0) for aa in seq[:max_len]]
        lengths.append(len(token_ids))
        if len(token_ids) < max_len:
            token_ids += [pad_token] * (max_len - len(token_ids))
        tokenized.append(token_ids)
    return torch.tensor(tokenized, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

class AminoAcidDataset(Dataset):
    def __init__(self, sequences, max_len):
        self.tokens, self.lengths = tokenize_and_pad(sequences, max_len, pad_token)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.lengths[idx]
    
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_layers=6, num_heads=8, max_len=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, hidden_dim))  # learnable positional encoding
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.time_embed = nn.Linear(1, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, t):
        B, L = x.shape
        x_embed = self.embedding(x) + self.pos_encoding[:, :L, :]
        t_embed = self.time_embed(t.unsqueeze(-1)).unsqueeze(1).expand(-1, L, -1)
        x_input = x_embed + t_embed
        out = self.transformer(x_input)
        return self.output_layer(out)
    

fasta_path = "training_sequences.fasta"
sequences = load_fasta_sequences(fasta_path)
dataset = AminoAcidDataset(sequences, max_seq_len)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = TransformerModel(vocab_size=vocab_size, max_len=max_seq_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=3.0))
loss_fn = MixturePathGeneralizedKL(path)
epsilon = 1e-3

iterations = 10001
print_every = 1000
steps = 0
start_time = time.time()

while steps < iterations:
    for x_1, lengths in dataloader:
        x_1 = x_1.to(device)
        B, L = x_1.shape
        x_0 = torch.randint(0, vocab_size - 1, (B, L), device=device)  # exclude pad_token from source
        t = torch.rand(B, device=device) * (1 - epsilon)

        xt_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        logits = model(xt_sample.x_t, xt_sample.t)
        loss = loss_fn(logits, x_1, xt_sample.x_t, xt_sample.t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    steps += 1
    if steps % print_every == 0:
        elapsed = time.time() - start_time
        print(f"| Iter {steps:6d} | {elapsed*1000/print_every:6.2f} ms/step | Loss {loss.item():8.3f}")
        start_time = time.time()

    if steps >= iterations:
                break
    
def generate_samples(model, path, num_samples=5, min_len=1, max_len=1024):
    model.eval()
    lens = torch.randint(min_len, max_len + 1, (num_samples,))
    init = [torch.randint(0, vocab_size, (L,)) for L in lens]
    x_init, lengths = pad_sequences(init)
    x_init = x_init.to(device)

    # Define softmax-wrapped model
    class WrappedModel(ModelWrapper):
        def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
            return torch.softmax(model(x, t), dim=-1)

    # Sampling final output
    solver = MixtureDiscreteEulerSolver(model=WrappedModel(model), path=path, vocabulary_size=vocab_size)
    samples = solver.sample(x_init=x_init, step_size=1.0 / 1024, verbose=True)

    # Decode final sequences
    decoded_final = []
    length_final = []
    for i, L in enumerate(lens):
        seq = samples[i][:L].cpu().tolist()
        aa_seq = "".join([idx_to_aa[idx] for idx in seq if idx != pad_token])
        decoded_final.append(aa_seq)
        length_final.append(len(aa_seq))

    # Intermediate states
    t_checkpoints = [0.1, 0.2, 0.5]
    intermediate_decoded = {}
    intermediate_lengths = {}

    for t_val in t_checkpoints:
        t = torch.full((x_init.size(0),), t_val, device=x_init.device)
        x_t = path.sample(t=t, x_0=x_init, x_1=samples).x_t
        decoded = []
        lengths = []
        for i, L in enumerate(lens):
            seq = x_t[i][:L].cpu().tolist()
            aa_seq = "".join([idx_to_aa[idx] for idx in seq if idx != pad_token])
            decoded.append(aa_seq)
            lengths.append(len(aa_seq))
        intermediate_decoded[f"t={t_val}"] = decoded
        intermediate_lengths[f"t={t_val}"] = lengths

    return decoded_final, length_final, intermediate_decoded, intermediate_lengths

def export_to_csv(final_seqs, final_lengths, mid_seqs_dict, mid_lengths_dict, filename="generated_sequences.csv"):
    data = {
        "sequence": final_seqs,
        "length": final_lengths,
    }

    for t_label in mid_seqs_dict:
        data[f"intermediate_{t_label}"] = mid_seqs_dict[t_label]
        data[f"length_{t_label}"] = mid_lengths_dict[t_label]

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved {len(final_seqs)} sequences to {filename}")

final_seqs, final_lengths, mid_seqs, mid_lengths = generate_samples(
    model, path, num_samples=1000, min_len=300, max_len=1024
)
export_to_csv(
    final_seqs, final_lengths,
    mid_seqs, mid_lengths,
    filename="samples/0413_1e3_n3.csv"
)