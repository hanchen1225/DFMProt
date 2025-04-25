import os
import pandas as pd
import requests
from tqdm import tqdm

input_folder = "./biogrid"  
output_fasta = "training_sequences.fasta"
uniprot_col_a = "SWISS-PROT Accessions Interactor A"
uniprot_col_b = "SWISS-PROT Accessions Interactor B"

def fetch_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    return response.text if response.ok else None

all_uniprot_ids = set()

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(input_folder, filename)
        df = pd.read_csv(filepath, sep="\t")

        for col in [uniprot_col_a, uniprot_col_b]:
            if col in df.columns:
                for entry in df[col].dropna():
                    ids = entry.split('|')
                    all_uniprot_ids.update(i.strip() for i in ids if i.strip())

print(f"🧬 Found {len(all_uniprot_ids)} unique UniProt IDs.")

with open(output_fasta, "w") as out_f:
    for uniprot_id in tqdm(all_uniprot_ids, desc="Fetching sequences"):
        seq = fetch_uniprot_sequence(uniprot_id)
        if seq:
            out_f.write(seq)

print(f"✅ Done! Sequences saved to '{output_fasta}'")