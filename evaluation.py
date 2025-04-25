import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy, wasserstein_distance
from Bio import SeqIO
import Levenshtein
import os

import umap

# Create output folder
os.makedirs("plots", exist_ok=True)

# Load real data (first 1024 amino acids per sequence)
def load_real_sequences(fasta_path):
    real_seqs = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        real_seqs.append(str(record.seq)[:1024])
    return real_seqs

# Load generated sequences
def load_generated_sequences(csv_path):
    df = pd.read_csv(csv_path)
    return df["intermediate_t=0.5"].tolist()

# Length distribution evaluation
def length_distribution_eval(real_seqs, gen_seqs, title="Length Distribution", output="plots/length_distribution.png"):
    real_lengths = [len(seq) for seq in real_seqs]
    gen_lengths = [len(seq) for seq in gen_seqs]
    
    plt.figure(figsize=(6, 4))
    sns.histplot(real_lengths, color="blue", label="Real", kde=False, stat="density", bins=50, alpha=0.5)
    sns.histplot(gen_lengths, color="red", label="Generated", kde=False, stat="density", bins=50, alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.xlabel("Sequence Length")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(output)
    plt.show()

    # KL Divergence
    real_counts = np.histogram(real_lengths, bins=range(0, max(real_lengths + gen_lengths)+2))[0] + 1e-8
    gen_counts = np.histogram(gen_lengths, bins=range(0, max(real_lengths + gen_lengths)+2))[0] + 1e-8
    real_probs = real_counts / real_counts.sum()
    gen_probs = gen_counts / gen_counts.sum()
    kl = entropy(real_probs, gen_probs)
    wd = wasserstein_distance(real_lengths, gen_lengths)
    print(f"[Length] KL divergence: {kl:.4f}, Wasserstein: {wd:.4f}")

# Amino acid frequency evaluation
def aa_frequency(seqs):
    aa_list = sorted("ACDEFGHIKLMNPQRSTVWY")
    total_counts = Counter()
    for seq in seqs:
        total_counts.update(seq)
    total = sum(total_counts.values())
    freqs = {aa: total_counts.get(aa, 0) / total for aa in aa_list}
    return freqs

def aa_frequency_eval(real_seqs, gen_seqs, title="Amino Acid Frequency", output="plots/aa_frequency.png"):
    real_freq = aa_frequency(real_seqs)
    gen_freq = aa_frequency(gen_seqs)

    aa = list(real_freq.keys())
    real_vals = [real_freq[a] for a in aa]
    gen_vals = [gen_freq[a] for a in aa]

    x = np.arange(len(aa))
    plt.figure(figsize=(8, 4))
    plt.bar(x - 0.2, real_vals, width=0.4, label="Real")
    plt.bar(x + 0.2, gen_vals, width=0.4, label="Generated")
    plt.xticks(x, aa)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.show()

    kl = entropy(real_vals, gen_vals)
    cosine_sim = cosine_similarity([real_vals], [gen_vals])[0,0]
    print(f"[AA Frequency] KL: {kl:.4f}, Cosine similarity: {cosine_sim:.4f}")

# N-gram evaluation
def ngram_freq(seqs, n):
    counter = Counter()
    total = 0
    for seq in seqs:
        ngrams = [seq[i:i+n] for i in range(len(seq)-n+1)]
        counter.update(ngrams)
        total += len(ngrams)
    return {k: v/total for k, v in counter.items()}

def ngram_eval(real_seqs, gen_seqs, title="N-gram KL Divergence", output="plots/ngram_kl.png"):
    bi_real = ngram_freq(real_seqs, 2)
    bi_gen = ngram_freq(gen_seqs, 2)
    tri_real = ngram_freq(real_seqs, 3)
    tri_gen = ngram_freq(gen_seqs, 3)

    def kl_from_freqs(f1, f2):
        all_keys = set(f1.keys()).union(f2.keys())
        p = np.array([f1.get(k, 1e-8) for k in all_keys])
        q = np.array([f2.get(k, 1e-8) for k in all_keys])
        return entropy(p, q)

    bi_kl = kl_from_freqs(bi_real, bi_gen)
    tri_kl = kl_from_freqs(tri_real, tri_gen)

    print(f"[N-gram KL] Bigram: {bi_kl:.4f}, Trigram: {tri_kl:.4f}")

# Diversity metrics

def diversity_eval(gen_seqs):
    uniq_ratio = len(set(gen_seqs)) / len(gen_seqs)
    
    # Sample to reduce compute (can be adjusted or removed for full eval)
    sampled_seqs = gen_seqs[:300]
    
    total_dist = 0
    count = 0
    for i in range(len(sampled_seqs)):
        for j in range(i + 1, len(sampled_seqs)):
            total_dist += Levenshtein.distance(sampled_seqs[i], sampled_seqs[j])
            count += 1
    
    avg_ed = total_dist / count if count > 0 else 0
    print(f"[Diversity] Unique sequences: {100 * uniq_ratio:.2f}%, Avg. Edit Distance (sampled): {avg_ed:.2f}")

# t-SNE and UMAP visualization
def embedding_plot(real_seqs, gen_seqs, method="tsne", title="Embedding of AA Frequencies", output="plots/embedding.png"):
    aa_list = sorted("ACDEFGHIKLMNPQRSTVWY")

    def seq_to_freq_vector(seq, aa_order):
        counter = Counter(seq)
        total = sum(counter.values())
        return [counter.get(aa, 0) / total if total > 0 else 0 for aa in aa_order]

    all_seqs = real_seqs + gen_seqs
    features = [seq_to_freq_vector(seq, aa_list) for seq in all_seqs]
    features = np.array(features)
    features = normalize(features)
    labels = ["Real"] * len(real_seqs) + ["Generated"] * len(gen_seqs)

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)

    embedding = reducer.fit_transform(features)
    df = pd.DataFrame(embedding, columns=["x", "y"])
    df["label"] = labels

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="x", y="y", hue="label", alpha=0.6)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output)
    plt.show()

# Main evaluation (n = 1)
real_seqs = load_real_sequences("training_sequences.fasta")
gen_seqs = load_generated_sequences("samples/0414_1e3_n1.csv")

length_distribution_eval(real_seqs, gen_seqs, title="Sequence Length Comparison (n = 1)", output="plots/n1_length_distribution.png")
aa_frequency_eval(real_seqs, gen_seqs, title="Amino Acid Usage (n = 1)", output="plots/n1_aa_frequency.png")
ngram_eval(real_seqs, gen_seqs, title="N-gram Divergence (n = 1)", output="plots/n1_ngram_kl.png")
diversity_eval(gen_seqs)
embedding_plot(real_seqs[:300], gen_seqs[:300], method="tsne", title="t-SNE of Amino Acid Frequency (n = 1)", output="plots/n1_tsne_plot.png")
embedding_plot(real_seqs[:300], gen_seqs[:300], method="umap", title="UMAP of Amino Acid Frequency (n = 1)", output="plots/n1_umap_plot.png")