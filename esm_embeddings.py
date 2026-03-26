import torch
import os
import pandas as pd
from transformers import EsmTokenizer, EsmModel
from tqdm import tqdm

INPUT_FILE = "bacillales_1385_3families_sequences.csv"            #<---- Change this
OUTPUT_FILE = "bacillales_1385_3families_embeddings.pt"           #<---- Change this
MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
BATCH_SIZE = 16

REPORT_EVERY = 5000
SAVE_EVERY = 5000

def get_embeddings():
    print(f"Loading model {MODEL_NAME}...")
    # Fetch the device that is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using device: CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"Model loaded successfully.")
    print(f"Reading sequences from {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df['gene_id'] = df['gene_id'].astype(str)
    # Checks for duplicates or NaNs
    df=df.dropna(subset=['sequence'])
    # Dictionary to store embeddings {'gene_id: embedding (Tensor)}
    if os.path.exists(OUTPUT_FILE):
        print(f"Resuming from checkpoint {OUTPUT_FILE}")
        embeddings_dict = torch.load(OUTPUT_FILE, map_location="cpu")
    else:
        embeddings_dict = {}

    done_ids = set(embeddings_dict.keys())
    pairs = [(gid, seq) for gid, seq in zip(df['gene_id'], df['sequence']) if gid not in done_ids]
    # Length-based batching
    pairs.sort(key=lambda x: len(x[1]))
    ids = [p[0] for p in pairs]
    sequences = [p[1] for p in pairs]
    print(f"Found {len(ids)} sequences.")

    print(f"Generating embeddings")
    processed = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(ids), BATCH_SIZE)):
            batch_ids = ids[i:i+BATCH_SIZE]
            batch_sequences = sequences[i:i+BATCH_SIZE]
            inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True,truncation=True,max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
             # --- Explanation of Masked Mean Pooling ---
            # The model outputs a tensor of shape [Batch_Size, Sequence_Length, Hidden_Dim].
            # However, sequences in a batch have different lengths and are padded with special tokens (zeros) to match the longest sequence.
            # We want a single vector representing the protein, but we must ignore these padding tokens to avoid skewing the average.
            # Masked Mean Pooling is a technique that allows us to ignore the padding tokens by masking them with zeros.

            # Get raw embeddings for all tokens (including padding)
            raw_embeddings = outputs.last_hidden_state
            # Get the attention mask (1 for tokens that are not padding, 0 for padding)
            attention_mask = inputs['attention_mask']
            # Apply mask: multiply emmbidings by the mask.
            # Real tokens*1 = unchanged.
            # Padding tokens*0=0
            masked_emmbedings=raw_embeddings * attention_mask.unsqueeze(-1)
            sum_emmbedings=masked_emmbedings.sum(dim=1)
            # Count the number of valid tokens
            token_counts=attention_mask.sum(dim=1).unsqueeze(-1)
            # Compute mean= sum / count
            token_counts=torch.clamp(token_counts,min=1e-9)
            mean_emmbedings=sum_emmbedings/token_counts
            for k , gene_id in enumerate(batch_ids):
                embeddings_dict[gene_id]=mean_emmbedings[k].cpu()
                processed += 1
                if processed % REPORT_EVERY == 0:
                    print(f"[INFO] Embedded {processed} / {len(ids)} sequences")
                if processed % SAVE_EVERY == 0:
                    print(f"[CHECKPOINT] Saving {len(embeddings_dict)} embeddings")
                    torch.save(embeddings_dict, OUTPUT_FILE)
            del outputs, raw_embeddings, masked_emmbedings, mean_emmbedings
        print(f"Saving to {OUTPUT_FILE}")
        # Saving as pytorch format
        torch.save(embeddings_dict,OUTPUT_FILE)
        print("Done!")

if __name__=='__main__':
    get_embeddings()
    