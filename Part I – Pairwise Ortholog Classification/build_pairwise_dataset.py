import numpy as np
import pandas as pd
import torch
from itertools import combinations
import random
from sklearn.model_selection import train_test_split
from collections import Counter

# --- FILE PATHS ---
OG_FILE  = "bacillales_1385_3families_geneID_to_ogID.csv"              #<---- Change this
EMBEDDINGS_FILE = "bacillales_1385_3families_embeddings.pt"            #<---- Change this

TRAIN_PAIRS = "train_pairs.csv"
VAL_PAIRS   = "val_pairs.csv"
TEST_PAIRS  = "test_pairs.csv"

# --- CONFIGURATION ---
POS_PER_OG = 50
NEG_RATIO = 1.0           # positives ≈ negatives
HARD_NEG_RATIO = 0.5      # 50% of negatives should be hard (in Train)
HARD_NEG_TOP_K = 200       # search space for hard negatives
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class DatasetLoader:
    def __init__(self, og_csv, embeddings_path):
        self.df = pd.read_csv(og_csv)
        self.df["gene_id"] = self.df["gene_id"].astype(str)
        self.df["orthogroup_id"] = self.df["og_id"].astype(str)

        print(f"Loading embeddings from {embeddings_path}...")
        raw_embeddings = torch.load(embeddings_path) 
        
        self.embeddings = {}
        for gene_id, tensor in raw_embeddings.items():
            #self.embeddings[str(gene_id)] = tensor.numpy()
            self.embeddings[str(gene_id)] = tensor.detach().cpu().numpy()

    def get_gene_to_og(self):
        return dict(zip(self.df.gene_id, self.df.orthogroup_id))

    def get_genes_by_og(self):
        return self.df.groupby("orthogroup_id")["gene_id"].apply(list).to_dict()

class OGSplitter:
    def __init__(self, gene_to_og, random_state=42):
        self.gene_to_og = gene_to_og
        self.random_state =random_state
        self.og_counts = Counter(gene_to_og.values())
        self.valid_ogs = [og for og, count in self.og_counts.items() if count >= 2]

    def split(self, train_size=0.8, val_size=0.1, test_size=0.1):
        train_ogs, temp_ogs = train_test_split(
            self.valid_ogs, train_size=train_size, random_state=self.random_state, shuffle=True
        )
        val_fraction = val_size / (val_size + test_size)
        val_ogs, test_ogs = train_test_split(
            temp_ogs, train_size=val_fraction, random_state=self.random_state, shuffle=True
        )
        return set(train_ogs), set(val_ogs), set(test_ogs)

class PairGenerator:
    def __init__(self, genes_by_og, embeddings, gene_to_og):
        self.genes_by_og = genes_by_og
        self.embeddings = embeddings
        self.gene_to_og = gene_to_og
        self.HARD_POS_THRESHOLD = 0.75 
        
        print("Normalizing embeddings...")
        for g in self.embeddings:
            norm = np.linalg.norm(self.embeddings[g])
            if norm > 0:
                self.embeddings[g] = self.embeddings[g] / norm

    def generate_pairs_for_ogs(self, og_set, mining_strategy="hard"):
        all_pairs = []
        pn_hp_hn=[]
        split_genes_set = set() 
        seen_pairs = set()

        target_mining_hard = POS_PER_OG // 2
        target_mining_random = POS_PER_OG - target_mining_hard
        POOL_FACTOR = 10 

        print(f"  > Generating pairs (Strategy: {mining_strategy})...")

        # POSITIVES
        for og in og_set:
            genes = [g for g in self.genes_by_og[og] if g in self.embeddings]
            split_genes_set.update(genes)
            
            if len(genes) < 2:
                continue
            
            # Generate Candidate Pool
            max_possible_pairs = (len(genes) * (len(genes) - 1)) // 2
            pool_size = POS_PER_OG * POOL_FACTOR
            candidate_pairs = []
            
            if max_possible_pairs <= pool_size:
                candidate_pairs = list(combinations(genes, 2))
            else:
                seen_cand = set()
                while len(candidate_pairs) < pool_size:
                    g1, g2 = random.sample(genes, 2)
                    pair = tuple(sorted((g1, g2)))
                    if pair not in seen_cand:
                        seen_cand.add(pair)
                        candidate_pairs.append(pair)

            # Compute Similarities
            pairs_with_scores = []
            for g1, g2 in candidate_pairs:
                sim = float(np.dot(self.embeddings[g1], self.embeddings[g2]))
                pairs_with_scores.append((g1, g2, sim))
            
            # Select Pairs
            selected_pairs = []
            if mining_strategy == "hard":
                # Sort: Lowest similarity first (Hardest)
                pairs_with_scores.sort(key=lambda x: x[2]) 
                mining_hard = pairs_with_scores[:target_mining_hard]
                remaining = pairs_with_scores[target_mining_hard:]
                
                mining_random = []
                if len(remaining) > 0:
                    needed = min(len(remaining), target_mining_random)
                    mining_random = random.sample(remaining, needed)
                selected_pairs = mining_hard + mining_random
            else:
                # Random strategy (Test/Val)
                needed = min(len(pairs_with_scores), POS_PER_OG)
                selected_pairs = random.sample(pairs_with_scores, needed)

            # Label and Add
            for g1, g2, sim in selected_pairs:
                # Dynamic Labeling: It is "Hard" if it is below threshold, regardless of strategy
                p_type = "hard_positive" if sim < self.HARD_POS_THRESHOLD else "soft_positive"
                
                pair_key = tuple(sorted((g1, g2)))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    all_pairs.append((g1, g2, og, og, 1))#, sim, p_type))
                    pn_hp_hn.append((sim,p_type))
        split_genes = list(split_genes_set)

        # NEGATIVES
        num_positives = len(all_pairs)
        total_negatives_needed = int(num_positives * NEG_RATIO)

        # Only mine hard negatives if strategy is "hard"
        if mining_strategy == "hard":
            target_hard_neg = int(total_negatives_needed * HARD_NEG_RATIO)
        else:
            target_hard_neg = 0
        
        print(f"  > Targets: Pos={num_positives}, Hard Neg={target_hard_neg} (Rest Random)")

        # Hard Negatives Loop 
        hard_neg_count = 0
        if target_hard_neg > 0 and len(split_genes) > 0:
            source_genes = random.sample(split_genes, min(target_hard_neg, len(split_genes)))
            for g1 in source_genes:
                og1 = self.gene_to_og[g1]
                vec1 = self.embeddings[g1]
                
                # Sample candidates
                candidates = random.sample(split_genes, min(len(split_genes), HARD_NEG_TOP_K))
                best_g2 = None
                best_sim = -1.0 
                
                for g2 in candidates:
                    if g1 == g2 or self.gene_to_og[g2] == og1: continue
                    sim = float(np.dot(vec1, self.embeddings[g2]))
                    if sim > best_sim:
                        best_sim = sim
                        best_g2 = g2
                
                if best_g2:
                    pair_key = tuple(sorted((g1, best_g2)))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        all_pairs.append((g1, best_g2, og1, self.gene_to_og[best_g2], 0))#, best_sim, "hard_negative"))
                        hard_neg_count += 1
                        pn_hp_hn.append((best_sim,"hard_negative"))

        # Random Negatives Loop 
        current_neg_count = hard_neg_count
        if len(split_genes) >= 2:
            while current_neg_count < total_negatives_needed:
                g1, g2 = random.sample(split_genes, 2)
                if self.gene_to_og[g1] != self.gene_to_og[g2]:
                    pair_key = tuple(sorted((g1, g2)))
                    if pair_key in seen_pairs: continue
                    
                    sim = float(np.dot(self.embeddings[g1], self.embeddings[g2]))
                    seen_pairs.add(pair_key)
                    all_pairs.append((g1, g2, self.gene_to_og[g1], self.gene_to_og[g2], 0))#, sim, "random_negative"))
                    current_neg_count += 1
                    pn_hp_hn.append((sim,"random_negative"))
        # PRINT STATS
        #counts = Counter([p[6] for p in all_pairs])
        counts=Counter(i[1] for i in pn_hp_hn)
        print("-" * 50)
        print(f"  STATS SUMMARY (Threshold < {self.HARD_POS_THRESHOLD})")
        print("-" * 50)
        print(f"  [+] Hard Positives:  {counts['hard_positive']} \t(Low sim, difficult)")
        print(f"  [+] Soft Positives:  {counts['soft_positive']} \t(High sim, easy)")
        print(f"  [-] Hard Negatives:  {counts['hard_negative']} \t(Mined high sim)")
        print(f"  [-] Random Negatives:{counts['random_negative']} \t(Randomly picked)")
        print("-" * 50)
        print(f"  TOTAL PAIRS:         {len(all_pairs)}")
        print("-" * 50)
        print("\n")

        return all_pairs

def build_split(name, ogs, generator, output_file, mining_strategy="hard"):
    print(f"Generating {name} pairs...")
    pairs = generator.generate_pairs_for_ogs(ogs, mining_strategy=mining_strategy)

    df = pd.DataFrame(pairs, columns=["gene1_id", "gene2_id", "orthogroup1_id", "orthogroup2_id", "label"])#, "similarity", "type"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_file, index=False)
    print(f"{name}: Saved to {output_file}")

if __name__ == "__main__":
    print("Loading data...")
    loader = DatasetLoader(OG_FILE, EMBEDDINGS_FILE)

    gene_to_og = loader.get_gene_to_og()
    genes_by_og = loader.get_genes_by_og()

    print("Splitting OGs...")
    splitter = OGSplitter(gene_to_og)
    train_ogs, val_ogs, test_ogs = splitter.split()

    generator = PairGenerator(genes_by_og, loader.embeddings, gene_to_og)

    # TRAIN: Use "hard" strategy to force difficult examples
    build_split("TRAIN", train_ogs, generator, TRAIN_PAIRS, mining_strategy="hard")
    
    # VAL: Use "random" (or "hard" if you prefer robust validation)
    build_split("VAL", val_ogs, generator, VAL_PAIRS, mining_strategy="random")
    
    # TEST: Use "random" for representative accuracy
    build_split("TEST", test_ogs, generator, TEST_PAIRS, mining_strategy="random")