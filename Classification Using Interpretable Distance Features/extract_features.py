import pandas as pd # type: ignore
import numpy as np # type: ignore
import Levenshtein # type: ignore
import zlib
from itertools import product

EMBEDDINGS_FILE = "bacillales_1385_3families_embeddings.pt"
SEQUENCES_FILE = "bacillales_1385_3families_sequences.csv"
PAIRS_FILES = {
    "train_pairs.csv": "train_pairs_features.csv",
    "val_pairs.csv":   "val_pairs_features.csv",
    "test_pairs.csv":  "test_pairs_features.csv",
}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
KMER_K = 3      # Size of k-mer (3 is standard for amino acids)


def pair_key(df):
    a = df["gene1_id"].astype(str)
    b = df["gene2_id"].astype(str)
    return pd.DataFrame({"p": a.where(a < b, b) + "||" + b.where(a < b, a)})

def normalized_levenshtein(s1,s2):
    """
    Computes Levenshtein distance normalized by the length of the longer sequence.
    Returns value between 0 (identical) and 1 (completely different).
    """
    denom = max(len(s1), len(s2))
    if denom == 0:
        return 0.0
    return Levenshtein.distance(s1, s2) / denom

#def geodesic_kmer_k_distances(s1,s2,)

def generate_kmer_map(k, alphabet):
    """Generates all k-mers and a mapping to indices."""
    kmers = [''.join(t) for t in product(alphabet, repeat=k)]
    kmer_to_idx = {km: i for i, km in enumerate(kmers)}
    return kmer_to_idx, len(kmers)

KMER_MAP, KMER_DIM = generate_kmer_map(KMER_K, AMINO_ACIDS)
print(f"Hellinger Config: K={KMER_K}, Dimensions={KMER_DIM}")

def get_kmer_prob_vector(seq, kmer_map, dim, k,alpha=1e-6):
    """
    Converts a sequence string into a normalized probability vector (numpy array).
    """
    counts = np.zeros(dim, dtype=np.float32)
    
    if len(seq) >= k:
        for i in range(len(seq) - k + 1):
            sub = seq[i : i+k]
            j=kmer_map.get(sub)
            if j is not None:
                counts[j] += 1.0
            
    # L1 Normalize (Sum to 1)
    counts += alpha
    counts /= counts.sum()
    return counts

def hellinger_distance(p, q):
    """
    Computes Hellinger distance between two probability vectors.
    Range: [0, 1]

    Formula: 1/sqrt(2) * EuclideanNorm( sqrt(p) - sqrt(q) )
    """
    sqrt_p = np.sqrt(p)
    sqrt_q = np.sqrt(q)
    # Euclidean distance between the square roots
    diff = sqrt_p - sqrt_q
    euclidean = np.sqrt(np.sum(diff ** 2))
    return euclidean / np.sqrt(2)

import zlib

def ncd_symmetric(s1: str, s2: str) -> float:
    """
    Computes Symmetric Normalized Compression Distance (NCD).
    
    Formula:
        NCD(x, y) = (C(xy) + C(yx) - 2 * min(C(x), C(y))) / (2 * max(C(x), C(y)))
    """
    if s1 == s2:
        return 0.0
    
    # Convert to bytes
    b1 = s1.encode('utf-8')
    b2 = s2.encode('utf-8')
    # Compute components (using level=9 for max compression)
    cx = len(zlib.compress(b1, level=9))
    cy = len(zlib.compress(b2, level=9))
    cxy = len(zlib.compress(b1 + b2, level=9))
    cyx = len(zlib.compress(b2 + b1, level=9))
    
    # Calculate NCD
    numerator = cxy + cyx - 2 * min(cx, cy)
    denominator = 2 * max(cx, cy)
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator

def process_split(name, df, seq_map, kmer_map, kmer_dim, k):
        print(f"\nProcessing {name} set ({len(df)} pairs)...")
        
        # Lists to store new features
        lev_dists = []
        hellinger_dists = []
        ncd_dists = []
        
        # Counters for missing data
        missing_count = 0
        
        total = len(df)
        
        # Iterate over rows using itertuples for speed
        for row in df.itertuples():
            # Get Gene IDs (ensure they are strings)
            g1, g2 = str(row.gene1_id), str(row.gene2_id)
            
            # Check if sequences exist
            if g1 not in seq_map or g2 not in seq_map:
                lev_dists.append(None)
                hellinger_dists.append(None)
                ncd_dists.append(None)
                missing_count += 1
                continue
            
            s1 = seq_map[g1]
            s2 = seq_map[g2]
            
            # 1. Levenshtein
            lev = normalized_levenshtein(s1, s2)
            lev_dists.append(lev)
            
            # 2. Hellinger (K-mer)
            p = get_kmer_prob_vector(s1, kmer_map, kmer_dim, k)
            q = get_kmer_prob_vector(s2, kmer_map, kmer_dim, k)
            hel = hellinger_distance(p, q)
            hellinger_dists.append(hel)
            
            # 3. NCD (Symmetric)
            ncd = ncd_symmetric(s1, s2)
            ncd_dists.append(ncd)
            
            # Progress print every 10%
            if row.Index % max(1, (total // 10)) == 0:
                print(f"  {row.Index}/{total} processed...")

        # Assign columns
        df["lev_dist"] = lev_dists
        df["hellinger_dist"] = hellinger_dists
        df["ncd_dist"] = ncd_dists
        
        # Drop rows where sequences were missing
        if missing_count > 0:
            print(f"  Warning: {missing_count} pairs dropped due to missing sequences.")
            df.dropna(inplace=True)
            
        return df


if __name__ == "__main__":
    train_df = pd.read_csv("train_pairs.csv")
    val_df   = pd.read_csv("val_pairs.csv")
    test_df  = pd.read_csv("test_pairs.csv")

    print(f"Train: \nTotal Pairs = {len(train_df)}")
    print("train labels:\n", train_df["label"].value_counts())
    print(f"Val: \nTotal Pairs = {len(val_df)}")
    print("val labels:\n", val_df["label"].value_counts())
    print(f"Test: \nTotal Pairs = {len(test_df)}")
    print("test labels:\n", test_df["label"].value_counts())
    # duplicates check (same pair repeated within split)
    print("train duplicate pairs:", pair_key(train_df).duplicated().sum())
    print("val duplicate pairs:", pair_key(val_df).duplicated().sum())
    print("test duplicate pairs:", pair_key(test_df).duplicated().sum())
    #Embeddings
    #embeddings = torch.load(EMBEDDINGS_FILE, map_location="cpu", weights_only=False)
    #embeddings = {str(k): v.float() for k, v in embeddings.items()}
    #print("num embeddings:", len(embeddings))

    # Load sequences
    seq_df = pd.read_csv(SEQUENCES_FILE)
    seq_map=dict(zip(seq_df['gene_id'].astype(str), seq_df['sequence']))
    print(f"Loaded {len(seq_map)} sequences.")
    # Process TRAIN
    train_df_feats = process_split("TRAIN", train_df, seq_map, KMER_MAP, KMER_DIM, KMER_K)
    train_df_feats.to_csv("train_pairs_features.csv", index=False)
    print("Saved train_pairs_features.csv")

    # Process VAL
    val_df_feats = process_split("VAL", val_df, seq_map, KMER_MAP, KMER_DIM, KMER_K)
    val_df_feats.to_csv("val_pairs_features.csv", index=False)
    print("Saved val_pairs_features.csv")

    # Process TEST
    test_df_feats = process_split("TEST", test_df, seq_map, KMER_MAP, KMER_DIM, KMER_K)
    test_df_feats.to_csv("test_pairs_features.csv", index=False)
    print("Saved test_pairs_features.csv")

    print("\nAll feature extraction complete!")