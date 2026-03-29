import pandas as pd
import random
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from collections import Counter


import markov_clustering as mcl
from scipy.sparse import csr_matrix



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parse_gene_ids(cell):
    s = str(cell).strip()
    s = s.strip("[]").replace("'", "").replace('"', "")
    return [p.strip() for p in s.split(",") if p.strip()]

def load_embeddings(path):
    print(f"Loading embeddings from {path} ...")
    emb = torch.load(path, map_location="cpu")

    if isinstance(emb, (list, tuple)) and len(emb) >= 1 and isinstance(emb[0], dict):
        emb = emb[0]

    if not isinstance(emb, dict):
        raise ValueError("Embeddings file is not a dict {gene_id: tensor}.")

    cleaned = {}
    for k, v in emb.items():
        gid = str(k)
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        if not torch.is_tensor(v):
            continue
        v = v.detach().cpu()
        if v.ndim == 2 and v.shape[0] == 1:
            v = v.squeeze(0)
        if v.ndim != 1:
            continue
        cleaned[gid] = v.float()

    if len(cleaned) == 0:
        raise ValueError("No valid embeddings found after cleaning.")

    d0 = next(iter(cleaned.values())).shape[0]
    print(f"Loaded {len(cleaned)} embeddings. Dim = {d0}")
    return cleaned

def group_indices_by_og(og_labels):
    """
    Convert og_labels into a dict
    """
    og_to_idx = defaultdict(list)
    for i, og in enumerate(og_labels):
        og_to_idx[str(og)].append(i)
    return og_to_idx

def sample_positive_pairs_for_one_og(idxs,cap_pairs,rng):
    """
    Sample up to cap_pairs unique positive pairs within one OG.
    idxs: indices of genes belonging to the same OG.
    """
    m = len(idxs)
    if m < 2:
        return []
    # Total possible unordered pairs
    possible = m * (m - 1) // 2
    target = min(cap_pairs, possible)
    pairs = set()
    # Random sampling without enumerating all pairs.
    attempts = 0
    max_attempts = 50 * target if target > 0 else 0
    while len(pairs) < target and attempts < max_attempts:
        a, b = rng.sample(idxs, 2)
        if a > b:
            a, b = b, a
        pairs.add((a, b))
        attempts += 1
    return list(pairs)

def sample_negative_pairs_across_ogs(og_to_idx,target_neg,rng,max_attempts=None):
    """
    Sample negative pairs (i, j) where i and j come from different OGs.
    We sample by:
      - choosing two different OGs at random
      - choosing one random member from each OG
    """
    ogs = list(og_to_idx.keys())
    if len(ogs) < 2 or target_neg <= 0:
        return []

    neg_pairs = set()
    attempts = 0
    if max_attempts is None:
        max_attempts = 80 * target_neg  

    while len(neg_pairs) < target_neg and attempts < max_attempts:
        og_a, og_b = rng.sample(ogs, 2)
        a = rng.choice(og_to_idx[og_a])
        b = rng.choice(og_to_idx[og_b])

        if a > b:
            a, b = b, a

        neg_pairs.add((a, b))
        attempts += 1

    return list(neg_pairs)

def build_edge_supervision_from_partition(
    og_labels,
    X=None,                      
    max_pos=30000,
    max_neg=30000,
    pos_per_group_cap=1500,
    seed=None,
    require_both_classes=True,
    hard_neg_frac=0.7,
    hard_k=50,
    device_for_hard="mps"
):
    rng = random.Random(seed) if seed is not None else random.Random()
    N = len(og_labels)
    if N < 2:
        return (torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32))

    og_to_idx = group_indices_by_og(og_labels)

    # --- positives ---
    pos_set = set()
    for og, idxs in og_to_idx.items():
        if len(idxs) >= 2:
            pos_set.update(sample_positive_pairs_for_one_og(idxs, pos_per_group_cap, rng))
    pos_pairs = list(pos_set)
    rng.shuffle(pos_pairs)
    pos_pairs = pos_pairs[:max_pos]

    target_neg = min(max_neg, len(pos_pairs))

    hard_target = int(target_neg * hard_neg_frac)
    rand_target = target_neg - hard_target

    hard_pairs = []
    if hard_target > 0:
        if X is None:
            raise ValueError("X must be provided to sample hard negatives.")
        hard_pairs = sample_hard_negatives_knn(
            X=X, og_labels=og_labels, target_neg=hard_target,
            k=hard_k, device=device_for_hard,
            seed=(seed or 0) + 999
        )

    rand_pairs = sample_negative_pairs_across_ogs(
        og_to_idx=og_to_idx,
        target_neg=rand_target,
        rng=rng
    )

    neg_pairs = list(set(hard_pairs + rand_pairs))
    if len(neg_pairs) < target_neg:
        extra = sample_negative_pairs_across_ogs(
            og_to_idx=og_to_idx,
            target_neg=(target_neg - len(neg_pairs)),
            rng=rng
        )
        neg_pairs = list(set(neg_pairs + extra))

    if require_both_classes and (len(pos_pairs) == 0 or len(neg_pairs) == 0):
        return (torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32))

    pairs = pos_pairs + neg_pairs[:target_neg]
    i_idx = torch.tensor([a for a, b in pairs], dtype=torch.long)
    j_idx = torch.tensor([b for a, b in pairs], dtype=torch.long)
    y = torch.tensor([1.0]*len(pos_pairs) + [0.0]*min(target_neg, len(neg_pairs)),
                     dtype=torch.float32)

    return i_idx, j_idx, y

def precompute_and_save_edges(
    sets_csv,
    mapping_csv,
    embeddings_path,
    out_dir="precomputed_edges_dev",
    max_pos=20000,
    max_neg=20000,
    pos_per_group_cap=1500,
    require_both_classes=True,
    base_seed=42,
    save_gene_ids=True,
):
    os.makedirs(out_dir, exist_ok=True)

    ds = GeneSetDataset(sets_csv, mapping_csv, embeddings_path)

    saved = 0
    skipped = 0

    for set_idx in range(len(ds)):
        gene_ids, X, og_labels = ds[set_idx]
        N = len(gene_ids)

        if N < 2:
            skipped += 1
            continue

        # deterministic per-set seed (so you can reproduce edges)
        seed = base_seed + set_idx

        i_idx, j_idx, y = build_edge_supervision_from_partition(
            og_labels=og_labels,
            X=X,
            max_pos=max_pos,
            max_neg=max_neg,
            pos_per_group_cap=pos_per_group_cap,
            seed=seed,
            require_both_classes=require_both_classes
        )

        # skip sets where we couldn't form both classes (or empty)
        if y.numel() == 0:
            skipped += 1
            continue

        payload = {
            "set_idx": set_idx,
            "N": N,
            "i_idx": i_idx.cpu(),
            "j_idx": j_idx.cpu(),
            "y": y.cpu(),
        }
        if save_gene_ids:
            payload["gene_ids"] = gene_ids

        out_path = os.path.join(out_dir, f"edges_set_{set_idx:05d}.pt")
        torch.save(payload, out_path)

        saved += 1
        if saved % 50 == 0:
            print(f"Saved {saved} sets... (skipped {skipped})")

    print(f"\nDONE. Saved={saved}, Skipped={skipped}, OutDir={out_dir}")
    return out_dir

@torch.no_grad()
def sample_hard_negatives_knn(X,og_labels,target_neg,k=50,device="cpu",seed=42):
    """
    Hard negatives: different OG but among k-nearest neighbors (cosine similarity).
    """
    rng = random.Random(seed)
    N = X.shape[0]
    if N < 2 or target_neg <= 0:
        return []

    Z = X.to(device)
    Z = F.normalize(Z, dim=1)

    S = Z @ Z.T
    S.fill_diagonal_(-1.0)

    og = [str(x) for x in og_labels]

    neg_pairs = set()
    topk = min(k, N - 1)
    vals, idxs = torch.topk(S, k=topk, dim=1)

    candidates = []
    for i in range(N):
        for j in idxs[i].tolist():
            if og[i] != og[j]:
                a, b = (i, j) if i < j else (j, i)
                candidates.append((a, b))

    rng.shuffle(candidates)

    for a, b in candidates:
        if (a, b) not in neg_pairs:
            neg_pairs.add((a, b))
            if len(neg_pairs) >= target_neg:
                break

    return list(neg_pairs)




# Dataset Class
class GeneSetDataset(Dataset):
    def __init__(self, sets_csv, mapping_csv, embeddings_path):
        self.sets_df = pd.read_csv(sets_csv)

        self.embeddings = load_embeddings(embeddings_path)

        mapping_df = pd.read_csv(mapping_csv)
        mapping_df["gene_id"] = mapping_df["gene_id"].astype(str).str.strip()
        self.gene_to_og = dict(zip(mapping_df["gene_id"], mapping_df["og_id"]))

    def __len__(self):
        return len(self.sets_df)

    def __getitem__(self, idx):
        gene_ids = [g.strip() for g in parse_gene_ids(self.sets_df.iloc[idx]["gene_ids"]) if g.strip()]
        gene_ids = [g for g in gene_ids if g in self.embeddings and g in self.gene_to_og]

        if len(gene_ids) < 2:
            return [], torch.empty((0, 480), dtype=torch.float32), []

        X = torch.stack([self.embeddings[g] for g in gene_ids], dim=0)  # float (480,)
        og_labels = [self.gene_to_og[g] for g in gene_ids]
        return gene_ids, X, og_labels


class EdgeFileSetDataset(Dataset):
    """
    returns X [N,D], i_idx [E], j_idx [E], y [E], og_labels [N], set_idx
    """
    def __init__(self, gene_set_ds: GeneSetDataset, edges_dir: str, set_indices=None):
        self.ds = gene_set_ds
        self.files = sorted(glob.glob(os.path.join(edges_dir, "edges_set_*.pt")))
        if not self.files:
            raise ValueError(f"No edge files in {edges_dir}")

        if set_indices is not None:
            keep = set(set_indices)
            filtered = []
            for f in self.files:
                payload = torch.load(f, map_location="cpu")
                if int(payload["set_idx"]) in keep:
                    filtered.append(f)
            self.files = sorted(filtered)

        if not self.files:
            raise ValueError("No edge files after filtering.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, k):
        payload = torch.load(self.files[k], map_location="cpu")
        set_idx = int(payload["set_idx"])

        gene_ids, X, og_labels_raw = self.ds[set_idx]   # X: [N,D]
        i_idx = payload["i_idx"].long()
        j_idx = payload["j_idx"].long()
        y     = payload["y"].float()

        N_saved = int(payload["N"])
        if X.shape[0] != N_saved:
            raise ValueError(
                f"Set {set_idx}: N mismatch ds={X.shape[0]} vs saved={N_saved}. "
                "Make sure previous steps use the SAME filtering of gene_ids."
            )
        if len(gene_ids) == 0:
            raise ValueError(f"Set {set_idx}: empty gene set after filtering.")

        # Encode OG strings to integers per set
        og_labels_raw = [str(og) for og in og_labels_raw]
        uniq_ogs = {og: idx for idx, og in enumerate(sorted(set(og_labels_raw)))}
        og_labels = torch.tensor([uniq_ogs[og] for og in og_labels_raw], dtype=torch.long)

        return X, i_idx, j_idx, y, og_labels, set_idx

def collate_sets(batch):
    return batch  # keep sets separate

@torch.no_grad()
def build_knn_edge_index(X, k=50, device="cpu", make_undirected=True,mode="mutual"):
    """
    Build a kNN graph from embeddings using cosine similarity.

    Modes:
    - "mutual": edge (i,j) exists only if both nodes select each other (AND)
    - "union" : edge (i,j) exists if at least one node selects the other (OR)

    Returns:
      edge_index: LongTensor [2, E] on `device`
        - if make_undirected=True: includes both directions (i->j and j->i)
        - if make_undirected=False: returns each mutual pair once (i<j)
    """
    N = X.shape[0]
    if N < 2:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    Z = X.to(device)
    Z = F.normalize(Z, dim=1)

    # cosine similarity
    S = Z @ Z.T
    S.fill_diagonal_(-1.0)

    topk = min(k, N - 1)
    _, knn = torch.topk(S, k=topk, dim=1)  # [N, topk], knn[i] are neighbors of i

    # Build boolean adjacency for kNN: A[i, j] = True if j in kNN(i)
    A = torch.zeros((N, N), dtype=torch.bool, device=device)
    row = torch.arange(N, device=device).unsqueeze(1).expand(N, topk).reshape(-1)
    col = knn.reshape(-1)
    A[row, col] = True

    
    if mode == "mutual":
        # Mutual condition: A[i,j] & A[j,i]
        M = A & A.T
    elif mode == "union":
        M = A | A.T
    else:
        raise ValueError("mode must be 'mutual' or 'union'")

    # Extract edges
    if make_undirected:
        src, dst = M.nonzero(as_tuple=True)   
        edge_index = torch.stack([src, dst], dim=0)
    else:
        # keep each mutual pair once (upper triangle)
        src, dst = torch.triu(M, diagonal=1).nonzero(as_tuple=True)
        edge_index = torch.stack([src, dst], dim=0)

    return edge_index

class SetEdgeSupervisionDataset(Dataset):
    """
    For each set idx:
      returns X [N,D], i_idx [E], j_idx [E], y [E], set_idx
    Edges are generated from the TRUE OG partition (supervised).
    """
    def __init__(
        self,
        gene_set_ds: GeneSetDataset,
        max_pos=20000,
        max_neg=20000,
        pos_per_group_cap=1500,
        base_seed=123,
        require_both_classes=True,
        hard_neg_frac=0.7,
        hard_k=50,
        device_for_hard="cpu",
    ):
        self.ds = gene_set_ds
        self.max_pos = max_pos
        self.max_neg = max_neg
        self.pos_per_group_cap = pos_per_group_cap
        self.base_seed = base_seed
        self.require_both_classes = require_both_classes
        self.hard_neg_frac = hard_neg_frac
        self.hard_k = hard_k
        self.device_for_hard = device_for_hard

        # Precompute which indices are usable
        self.valid_indices = []
        for idx in range(len(self.ds)):
            gene_ids, X, og = self.ds[idx]
            if len(gene_ids) >= 2 and len(og) >= 2:
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, k):
        set_idx = self.valid_indices[k]
        gene_ids, X, og_labels = self.ds[set_idx]

        seed = self.base_seed + set_idx
        i_idx, j_idx, y = build_edge_supervision_from_partition(
            og_labels=og_labels,
            X=X,
            max_pos=self.max_pos,
            max_neg=self.max_neg,
            pos_per_group_cap=self.pos_per_group_cap,
            seed=seed,
            require_both_classes=self.require_both_classes,
            hard_neg_frac=self.hard_neg_frac,
            hard_k=self.hard_k,
            device_for_hard=self.device_for_hard,
        )

        return X, i_idx, j_idx, y, set_idx

def sample_triplets_from_og_labels(og_labels, max_triplets=256):
    """
    og_labels: 1D tensor/list of length N
    returns anchor_idx, pos_idx, neg_idx or (None, None, None)
    """
    if torch.is_tensor(og_labels):
        og_labels = og_labels.detach().cpu().tolist()

    og_to_nodes = {}
    for idx, og in enumerate(og_labels):
        og_to_nodes.setdefault(str(og), []).append(idx)

    all_nodes = list(range(len(og_labels)))

    anchors, positives, negatives = [], [], []

    for og, members in og_to_nodes.items():
        if len(members) < 2:
            continue

        neg_pool = [x for x in all_nodes if str(og_labels[x]) != og]
        if not neg_pool:
            continue

        for a in members:
            pos_candidates = [x for x in members if x != a]
            if not pos_candidates:
                continue

            p = random.choice(pos_candidates)
            n = random.choice(neg_pool)

            anchors.append(a)
            positives.append(p)
            negatives.append(n)

    if len(anchors) == 0:
        return None, None, None

    if len(anchors) > max_triplets:
        chosen = random.sample(range(len(anchors)), max_triplets)
        anchors   = [anchors[k] for k in chosen]
        positives = [positives[k] for k in chosen]
        negatives = [negatives[k] for k in chosen]

    return (
        torch.tensor(anchors, dtype=torch.long),
        torch.tensor(positives, dtype=torch.long),
        torch.tensor(negatives, dtype=torch.long),
    )

@torch.no_grad()
def sample_hard_triplets(H, og_labels, max_triplets=256, max_anchors=64):
    """
    Hard-triplet mining:
    - sample only a subset of anchors
    - compute distances only from each chosen anchor to all nodes
    - no full NxN distance matrix
    """
    if torch.is_tensor(og_labels):
        og_labels = og_labels.detach().to(H.device)
    else:
        og_labels = torch.tensor(og_labels, device=H.device)

    N = H.size(0)
    if N < 3:
        return None, None, None

    Hn = F.normalize(H.detach(), p=2, dim=1)

    anchor_pool = list(range(N))
    if len(anchor_pool) > max_anchors:
        anchor_pool = random.sample(anchor_pool, max_anchors)

    anchors, positives, negatives = [], [], []

    for a in anchor_pool:
        same = (og_labels == og_labels[a]).nonzero(as_tuple=True)[0]
        diff = (og_labels != og_labels[a]).nonzero(as_tuple=True)[0]

        same = same[same != a]
        if same.numel() == 0 or diff.numel() == 0:
            continue

        dists = torch.norm(Hn[a].unsqueeze(0) - Hn, p=2, dim=1)

        p = same[torch.argmax(dists[same])]
        n = diff[torch.argmin(dists[diff])]

        anchors.append(a)
        positives.append(int(p.item()))
        negatives.append(int(n.item()))

    if len(anchors) == 0:
        return None, None, None

    if len(anchors) > max_triplets:
        chosen = random.sample(range(len(anchors)), max_triplets)
        anchors   = [anchors[k] for k in chosen]
        positives = [positives[k] for k in chosen]
        negatives = [negatives[k] for k in chosen]

    return (
        torch.tensor(anchors, dtype=torch.long, device=H.device),
        torch.tensor(positives, dtype=torch.long, device=H.device),
        torch.tensor(negatives, dtype=torch.long, device=H.device),
    )

def compute_dual_loss(H, logits, y, og_labels, lambda_triplet=0.2, triplet_margin=0.2, max_triplets=256):
    """
    H: [N, d] node embeddings
    logits: [M] pair logits
    y: [M] pair labels
    og_labels: [N] OG id per node
    """
    bce = F.binary_cross_entropy_with_logits(logits, y.float())

    a, p, n = sample_hard_triplets(H, og_labels, max_triplets=max_triplets)

    if a is None:
        trip = torch.tensor(0.0, device=H.device)
    else:
        a = a.to(H.device)
        p = p.to(H.device)
        n = n.to(H.device)

        triplet_fn = nn.TripletMarginLoss(margin=triplet_margin, p=2)
        trip = triplet_fn(H[a], H[p], H[n])

    total = bce + lambda_triplet * trip
    return total, bce.detach(), trip.detach()

class SAGEBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_nei  = nn.Linear(in_dim, out_dim)
        self.drop     = nn.Dropout(dropout)

    def forward(self, h, edge_index):
        if edge_index.numel() == 0:
            return self.drop(F.relu(self.lin_self(h)))

        src, dst = edge_index[0], edge_index[1]
        msgs = h[src]  

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msgs)  

        deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(1)

        mean = agg / deg
        out = self.lin_self(h) + self.lin_nei(mean)
        out = self.drop(F.relu(out))
        return out
    
class SetGNNEncoder(nn.Module):
    def __init__(self, in_dim=480, hidden=256, layers=4, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([SAGEBlock(hidden, hidden, dropout) for _ in range(layers)])
        self.norms  = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

    def forward(self, X, edge_index):
        h = F.relu(self.inp(X))
        for blk, ln in zip(self.blocks, self.norms):
            h = ln(blk(h, edge_index))
        h=F.normalize(h, p=2, dim=1)
        return h

class EdgeHead(nn.Module):
    def __init__(self, h_dim=256, hidden=128, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*h_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, hi, hj):
        absdiff = (hi - hj).abs()
        had     = hi * hj
        feat = torch.cat([absdiff, had], dim=1)
        return self.mlp(feat).squeeze(1)  # logits

class SetEdgeModel(nn.Module):
    """
    This is f_theta.
    """
    def __init__(self, in_dim=480, h_dim=256, gnn_layers=3, dropout=0.1):
        super().__init__()
        self.enc  = SetGNNEncoder(in_dim=in_dim, hidden=h_dim, layers=gnn_layers, dropout=dropout)
        self.head = EdgeHead(h_dim=h_dim, hidden=128, dropout=dropout)

    def forward(self, X, edge_index, i_idx, j_idx):
        H = self.enc(X, edge_index)          
        hi, hj = H[i_idx], H[j_idx]          
        logits = self.head(hi, hj)           
        return H,logits

@torch.no_grad()
def eval_loss_acc(model, loader, device, knn_k, thr=0.65, lambda_triplet=0.3, triplet_margin=0.2):
    model.eval()
    tot_loss, tot_bce, tot_trip, tot_n = 0.0, 0.0, 0.0, 0
    correct = 0

    for batch in loader:
        for item in batch:
            X, i_idx, j_idx, y, og_labels, set_idx = item

            X = X.to(device)
            i_idx = i_idx.to(device)
            j_idx = j_idx.to(device)
            y = y.to(device)
            og_labels = og_labels.to(device)

            edge_index = build_knn_edge_index(X, k=knn_k, device=device)
            H, logits = model(X, edge_index, i_idx, j_idx)

            loss, bce, trip = compute_dual_loss(
                H, logits, y, og_labels,
                lambda_triplet=lambda_triplet,
                triplet_margin=triplet_margin
            )

            probs = torch.sigmoid(logits)
            pred = (probs >= thr).float()
            correct += int((pred == y).sum().item())

            tot_loss += float(loss.item()) * y.numel()
            tot_bce  += float(bce.item()) * y.numel()
            tot_trip += float(trip.item()) * y.numel()
            tot_n += y.numel()

    avg_loss = tot_loss / max(1, tot_n)
    avg_bce  = tot_bce / max(1, tot_n)
    avg_trip = tot_trip / max(1, tot_n)
    acc = correct / max(1, tot_n)
    return avg_loss, avg_bce, avg_trip, acc

def train_set_model(
    sets_csv,
    mapping_csv,
    embeddings_path,
    edges_dir="precomputed_edges_dev",
    out_path="set_gnn_edge_model.pt",
    seed=42,
    epochs=15,
    lr=1e-3,
    weight_decay=1e-4,
    val_frac=0.2,
    knn_k=50,
    batch_sets=1,
    device=None,
    lambda_triplet=0.4,
    triplet_margin=0.2,
):
    """
    Trains SetEdgeModel with:
        total_loss = BCE + lambda_triplet * TripletLoss

    Assumes each item from the loader is:
        (X, i_idx, j_idx, y, og_labels, set_idx)

    Where:
        X         : [N, 480] node/gene embeddings for one set
        i_idx,j_idx : indices of supervised training pairs
        y         : pair labels (0/1)
        og_labels : OG id per node in X, already encoded as integers
        set_idx   : set id
    """
    set_seed(seed)

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    device = torch.device(device)
    print("Device:", device)

    gs = GeneSetDataset(sets_csv, mapping_csv, embeddings_path)

    # Determine which set_idx have edge files
    edge_files = sorted(glob.glob(os.path.join(edges_dir, "edges_set_*.pt")))
    set_ids = []
    for f in edge_files:
        payload = torch.load(f, map_location="cpu")
        set_ids.append(int(payload["set_idx"]))
    set_ids = sorted(set(set_ids))
    random.Random(seed).shuffle(set_ids)

    n_val = max(1, int(len(set_ids) * val_frac))
    val_ids = set(set_ids[:n_val])
    train_ids = [s for s in set_ids if s not in val_ids]
    print(f"Sets with edges: {len(set_ids)} | train={len(train_ids)} | val={len(val_ids)}")

    train_ds = EdgeFileSetDataset(gs, edges_dir, set_indices=train_ids)
    val_ds   = EdgeFileSetDataset(gs, edges_dir, set_indices=val_ids)

    train_loader = DataLoader(train_ds, batch_size=batch_sets, shuffle=True, collate_fn=collate_sets)
    val_loader   = DataLoader(val_ds, batch_size=batch_sets, shuffle=False, collate_fn=collate_sets)

    model = SetEdgeModel(in_dim=480, h_dim=256, gnn_layers=3, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")

    train_losses, val_losses = [], []
    train_bces, val_bces = [], []
    train_trips, val_trips = [], []
    train_accs, val_accs = [], []

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss, tot_bce, tot_trip, tot_n = 0.0, 0.0, 0.0, 0

        for batch in train_loader:
            for item in batch:
                X, i_idx, j_idx, y, og_labels, set_idx = item

                X = X.to(device)
                i_idx = i_idx.to(device)
                j_idx = j_idx.to(device)
                y = y.to(device)
                og_labels = og_labels.to(device)

                edge_index = build_knn_edge_index(X, k=knn_k, device=device)

                H, logits = model(X, edge_index, i_idx, j_idx)

                loss, bce_loss, trip_loss = compute_dual_loss(
                    H=H,
                    logits=logits,
                    y=y,
                    og_labels=og_labels,
                    lambda_triplet=lambda_triplet,
                    triplet_margin=triplet_margin,
                    max_triplets=256,
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                tot_loss += float(loss.item()) * y.numel()
                tot_bce  += float(bce_loss.item()) * y.numel()
                tot_trip += float(trip_loss.item()) * y.numel()
                tot_n += y.numel()

        train_loss_epoch = tot_loss / max(1, tot_n)
        train_bce_epoch  = tot_bce / max(1, tot_n)
        train_trip_epoch = tot_trip / max(1, tot_n)

        val_loss, val_bce, val_trip, val_acc = eval_loss_acc(
            model=model,
            loader=val_loader,
            device=device,
            knn_k=knn_k,
            thr=0.65,
            lambda_triplet=lambda_triplet,
            triplet_margin=triplet_margin,
        )

        train_loss_eval, train_bce_eval, train_trip_eval, train_acc = eval_loss_acc(
            model=model,
            loader=train_loader,
            device=device,
            knn_k=knn_k,
            thr=0.65,
            lambda_triplet=lambda_triplet,
            triplet_margin=triplet_margin,
        )

        train_losses.append(train_loss_eval)
        val_losses.append(val_loss)
        train_bces.append(train_bce_eval)
        val_bces.append(val_bce)
        train_trips.append(train_trip_eval)
        val_trips.append(val_trip)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {ep:02d} | "
            f"train_loss={train_loss_eval:.4f} (bce={train_bce_eval:.4f}, trip={train_trip_eval:.4f}), "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} (bce={val_bce:.4f}, trip={val_trip:.4f}), "
            f"val_acc={val_acc:.4f}"
        )

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "in_dim": 480,
                    "h_dim": 256,
                    "gnn_layers": 3,
                    "knn_k": knn_k,
                    "best_val_loss": best_val,
                    "thr": 0.65,
                    "lambda_triplet": lambda_triplet,
                    "triplet_margin": triplet_margin,
                    "train_loss": train_loss_eval,
                    "val_loss": val_loss,
                    "train_bce": train_bce_eval,
                    "val_bce": val_bce,
                    "train_triplet": train_trip_eval,
                    "val_triplet": val_trip,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                },
                out_path,
            )
            print("  saved best ->", out_path)

        # Save checkpoint every 5 epochs
        if ep % 5 == 0:
            ckpt_path = out_path.replace(".pt", f"_epoch{ep:03d}.pt")
            torch.save(
                {
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "in_dim": 480,
                    "h_dim": 256,
                    "gnn_layers": 3,
                    "knn_k": knn_k,
                    "thr": 0.65,
                    "lambda_triplet": lambda_triplet,
                    "triplet_margin": triplet_margin,
                    "train_loss": train_loss_eval,
                    "val_loss": val_loss,
                    "train_bce": train_bce_eval,
                    "val_bce": val_bce,
                    "train_triplet": train_trip_eval,
                    "val_triplet": val_trip,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "best_val_loss": best_val,
                },
                ckpt_path,
            )
            print(f"  checkpoint saved -> {ckpt_path}")

    # Curves
    plt.figure()
    plt.plot(train_losses, label="train_total")
    plt.plot(val_losses, label="val_total")
    plt.xlabel("epoch")
    plt.ylabel("total loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_total_curve.png", dpi=200)
    plt.show()

    plt.figure()
    plt.plot(train_bces, label="train_bce")
    plt.plot(val_bces, label="val_bce")
    plt.xlabel("epoch")
    plt.ylabel("BCE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_bce_curve.png", dpi=200)
    plt.show()

    plt.figure()
    plt.plot(train_trips, label="train_triplet")
    plt.plot(val_trips, label="val_triplet")
    plt.xlabel("epoch")
    plt.ylabel("triplet loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_triplet_curve.png", dpi=200)
    plt.show()

    plt.figure()
    plt.plot(train_accs, label="train_acc")
    plt.plot(val_accs, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("acc_curve.png", dpi=200)
    plt.show()

    print("Saved plots: loss_total_curve.png, loss_bce_curve.png, loss_triplet_curve.png, acc_curve.png")
    return out_path

@torch.no_grad()
def predict_partition_for_set(model, X, knn_k=50, thr=0.65, device="cpu"):
    model.eval()
    X = X.to(device)

    edge_index = build_knn_edge_index(X, k=knn_k, device=device)
    H = model.enc(X, edge_index)

    src, dst = edge_index
    probs = torch.sigmoid(model.head(H[src], H[dst]))

    edges = [(a, b) for a, b, p in zip(src.tolist(), dst.tolist(), probs.tolist()) if p >= thr]

    # undirected adjacency
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    N = X.size(0)
    visited = [False] * N
    components = []
    for i in range(N):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        components.append(comp)

    return components

def evaluate_dev_partitions(
    model,
    dataset,
    set_indices,
    knn_k=50,
    thresholds=(0.4, 0.5, 0.6),
    device="cpu"
):
    results = []

    for thr in thresholds:
        og_errs = []
        singleton_prec = []
        singleton_rec = []

        for idx in set_indices:
            gene_ids, X, true_og = dataset[idx]
            if len(X) < 2:
                continue

            comps = predict_partition_for_set(
                model, X, knn_k=knn_k, thr=thr, device=device
            )

            # true OG structure
            og_to_true = group_indices_by_og(true_og)
            true_num_ogs = len(og_to_true)
            pred_num_ogs = len(comps)

            og_errs.append(abs(pred_num_ogs - true_num_ogs))

            true_singletons = {i for i, og in enumerate(true_og)
                               if len(og_to_true[str(og)]) == 1}
            pred_singletons = {c[0] for c in comps if len(c) == 1}

            if pred_singletons:
                singleton_prec.append(
                    len(pred_singletons & true_singletons) / len(pred_singletons)
                )
            if true_singletons:
                singleton_rec.append(
                    len(pred_singletons & true_singletons) / len(true_singletons)
                )

        results.append({
            "thr": thr,
            "mean_og_error": np.mean(og_errs),
            "singleton_precision": np.mean(singleton_prec),
            "singleton_recall": np.mean(singleton_rec),
        })

    return results

@torch.no_grad()
def predict_ogs_for_set(
    model,
    gene_ids,
    X,
    knn_k=50,
    thr=0.65,
    device="cpu",
):
    comps = predict_partition_for_set(model, X, knn_k=knn_k, thr=thr, device=device)

    ogs = []
    for comp in comps:
        ogs.append([gene_ids[i] for i in comp])

    K = len(ogs)
    return K, ogs

def load_trained_model(model_path, device="cpu"):
    ckpt = torch.load(model_path, map_location="cpu")
    model = SetEdgeModel(
        in_dim=ckpt.get("in_dim", 480),
        h_dim=ckpt.get("h_dim", 256),
        gnn_layers=ckpt.get("gnn_layers", 3),
        dropout=0.1
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt

def get_train_val_ids(edges_dir, seed=42, val_frac=0.2):
    edge_files = sorted(glob.glob(os.path.join(edges_dir, "edges_set_*.pt")))
    assert len(edge_files) > 0

    set_ids = []
    for f in edge_files:
        payload = torch.load(f, map_location="cpu")
        set_ids.append(int(payload["set_idx"]))
    set_ids = sorted(set(set_ids))

    rng = random.Random(seed)
    rng.shuffle(set_ids)

    n_val = max(1, int(len(set_ids) * val_frac))
    val_ids = set(set_ids[:n_val])
    train_ids = [s for s in set_ids if s not in val_ids]
    return train_ids, val_ids

@torch.no_grad()
def predict_partition_for_set_mcl(model, X, knn_k=150, thr=0.85, inflation=1.8, device="cpu"):
    model.eval()
    X = X.to(device)
    N = X.size(0)
    if N==0:
        return []
    if N==1:
        return [[0]]

    # Build the initial graph (Topological Context)
    edge_index = build_knn_edge_index(X, k=knn_k, device=device,make_undirected=True,mode="union")
    if edge_index.numel()==0:
        return [[i] for i in range(N)]
    
    # Get GNN Refined Embeddings
    H = model.enc(X, edge_index)

    # Predict Edge Probabilities only for the kNN edges
    src, dst = edge_index
    pair_set=set()
    for a, b in zip(src.tolist(), dst.tolist()):
        if a == b:
            continue
        i, j = (a, b) if a < b else (b, a)
        pair_set.add((i, j))

    pairs = sorted(pair_set)
    if not pairs:
        return [[i] for i in range(N)]

    i_idx = torch.tensor([i for i, j in pairs], dtype=torch.long, device=device)
    j_idx = torch.tensor([j for i, j in pairs], dtype=torch.long, device=device)
    logits_ij = model.head(H[i_idx], H[j_idx])
    logits_ji = model.head(H[j_idx], H[i_idx])

    probs_ij = torch.sigmoid(logits_ij)
    probs_ji = torch.sigmoid(logits_ji) 
    # symmetric probability per undirected pair
    probs = 0.5 * (probs_ij + probs_ji)
    probs = probs.cpu().numpy()
    rows, cols, data = [], [], []
    for (i, j), p in zip(pairs, probs):
        if p >= thr:
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([float(p), float(p)])
    # Add self-loops
    for i in range(N):
        rows.append(i)
        cols.append(i)
        data.append(1.0)

    # Create a Sparse Adjacency Matrix for MCL
    # We use the probabilities as weights for the Markov transitions
    adj_sparse = csr_matrix((data, (rows, cols)), shape=(N, N))

    # Run Markov Clustering
    # result is a matrix where clusters are represented by non-zero columns
    result = mcl.run_mcl(adj_sparse, inflation=inflation)
    clusters = mcl.get_clusters(result)
    # Convert to list of lists (list of OGs)
    components = [list(c) for c in clusters]
    return components

def evaluate_set_clustering_fast(gene_ids, true_ogs, pred_clusters):
    N = len(gene_ids)

    # map gene index -> predicted cluster id
    pred_label = [-1] * N
    for cid, cluster in enumerate(pred_clusters):
        for i in cluster:
            pred_label[i] = cid

    # true clusters
    true_groups = defaultdict(list)
    for i, og in enumerate(true_ogs):
        true_groups[str(og)].append(i)
    true_clusters = list(true_groups.values())

    # counts
    true_num_ogs = len(true_clusters)
    pred_num_ogs = len(pred_clusters)
    abs_count_error = abs(pred_num_ogs - true_num_ogs)

    true_singletons = sum(1 for c in true_clusters if len(c) == 1)
    pred_singletons = sum(1 for c in pred_clusters if len(c) == 1)

    # purity-style accuracy: for each predicted cluster, count majority true OG
    purity_correct = 0
    for cluster in pred_clusters:
        labs = [str(true_ogs[i]) for i in cluster]
        if labs:
            purity_correct += Counter(labs).most_common(1)[0][1]
    purity_accuracy = purity_correct / N if N > 0 else 1.0

    # coverage-style accuracy: for each true OG, count best overlapping predicted cluster
    coverage_correct = 0
    for true_cluster in true_clusters:
        pred_counts = Counter(pred_label[i] for i in true_cluster)
        if pred_counts:
            coverage_correct += pred_counts.most_common(1)[0][1]
    coverage_accuracy = coverage_correct / N if N > 0 else 1.0

    # singleton precision / recall
    true_singleton_genes = {c[0] for c in true_clusters if len(c) == 1}
    pred_singleton_genes = {c[0] for c in pred_clusters if len(c) == 1}

    singleton_tp = len(true_singleton_genes & pred_singleton_genes)
    singleton_precision = singleton_tp / len(pred_singleton_genes) if pred_singleton_genes else 0.0
    singleton_recall = singleton_tp / len(true_singleton_genes) if true_singleton_genes else 0.0

    return {
        "genes": N,
        "true_num_ogs": true_num_ogs,
        "pred_num_ogs": pred_num_ogs,
        "abs_count_error": abs_count_error,
        "true_singletons": true_singletons,
        "pred_singletons": pred_singletons,
        "purity_accuracy": purity_accuracy,
        "coverage_accuracy": coverage_accuracy,
        "singleton_precision": singleton_precision,
        "singleton_recall": singleton_recall,
    }

def og_coverage_metrics(true_og, pred_clusters):
    """
    Compare true OGs to predicted clusters.

    Returns:
        mean_og_coverage:
            for each true OG T, max_P |T ∩ P| / |T|
        exact_og_accuracy:
            fraction of true OGs exactly recovered by some predicted cluster
        mean_og_iou:
            for each true OG T, max_P |T ∩ P| / |T ∪ P|
    """
    og_to_true = group_indices_by_og(true_og)

    true_clusters = [set(v) for v in og_to_true.values()]
    pred_clusters = [set(c) for c in pred_clusters]

    if len(true_clusters) == 0:
        return {
            "mean_og_coverage": 0.0,
            "exact_og_accuracy": 0.0,
            "mean_og_iou": 0.0,
        }

    coverages = []
    exact_hits = []
    ious = []

    for T in true_clusters:
        best_cov = 0.0
        best_iou = 0.0
        exact = 0.0

        for P in pred_clusters:
            inter = len(T & P)
            union = len(T | P)

            cov = inter / len(T) if len(T) > 0 else 0.0
            iou = inter / union if union > 0 else 0.0

            if cov > best_cov:
                best_cov = cov
            if iou > best_iou:
                best_iou = iou
            if T == P:
                exact = 1.0

        coverages.append(best_cov)
        ious.append(best_iou)
        exact_hits.append(exact)

    return {
        "mean_og_coverage": float(np.mean(coverages)),
        "exact_og_accuracy": float(np.mean(exact_hits)),
        "mean_og_iou": float(np.mean(ious)),
    }

def partitions_mcl(
    model,
    dataset,
    set_indices,
    knn_k=150,
    thresholds=[0.75, 0.80, 0.85, 0.90],
    inflations=[1.4, 1.6, 1.8, 2.0],
    device="cpu"
):
    results = []
    per_set_results = []

    total_sets = len(set_indices)

    for thr in thresholds:
        for inflation in inflations:
            print(f"\n=== Running thr={thr}, inflation={inflation} ===")

            og_errs = []
            singleton_prec = []
            singleton_rec = []
            og_coverages = []
            exact_og_accuracies = []
            og_ious = []

            for i, idx in enumerate(set_indices):
                if i % 10 == 0:
                    print(f"[thr={thr}, infl={inflation}] Set {i+1}/{total_sets} (idx={idx})")

                gene_ids, X, true_og = dataset[idx]

                if len(X) == 0:
                    continue

                comps = predict_partition_for_set_mcl(
                    model=model,
                    X=X,
                    knn_k=knn_k,
                    thr=thr,
                    inflation=inflation,
                    device=device
                )

                og_to_true = group_indices_by_og(true_og)
                true_num_ogs = len(og_to_true)
                pred_num_ogs = len(comps)
                og_error = abs(pred_num_ogs - true_num_ogs)
                og_errs.append(og_error)

                true_singletons = {
                    members[0] for members in og_to_true.values() if len(members) == 1
                }
                pred_singletons = {
                    c[0] for c in comps if len(c) == 1
                }

                sp = (
                    len(pred_singletons & true_singletons) / len(pred_singletons)
                    if len(pred_singletons) > 0 else 0.0
                )
                sr = (
                    len(pred_singletons & true_singletons) / len(true_singletons)
                    if len(true_singletons) > 0 else 0.0
                )

                singleton_prec.append(sp)
                singleton_rec.append(sr)

                og_metrics = og_coverage_metrics(true_og, comps)
                og_coverages.append(og_metrics["mean_og_coverage"])
                exact_og_accuracies.append(og_metrics["exact_og_accuracy"])
                og_ious.append(og_metrics["mean_og_iou"])

                per_set_results.append({
                    "set_idx": idx,
                    "thr": thr,
                    "inflation": inflation,
                    "num_genes": len(gene_ids),
                    "true_num_ogs": true_num_ogs,
                    "pred_num_ogs": pred_num_ogs,
                    "og_error": og_error,
                    "singleton_precision": sp,
                    "singleton_recall": sr,
                    "mean_og_coverage": og_metrics["mean_og_coverage"],
                    "exact_og_accuracy": og_metrics["exact_og_accuracy"],
                    "mean_og_iou": og_metrics["mean_og_iou"],
                })

            print(f"Finished thr={thr}, inflation={inflation}")

            results.append({
                "thr": thr,
                "inflation": inflation,
                "mean_og_error": float(np.mean(og_errs)) if og_errs else None,
                "singleton_precision": float(np.mean(singleton_prec)) if singleton_prec else None,
                "singleton_recall": float(np.mean(singleton_rec)) if singleton_rec else None,
                "mean_og_coverage": float(np.mean(og_coverages)) if og_coverages else None,
                "exact_og_accuracy": float(np.mean(exact_og_accuracies)) if exact_og_accuracies else None,
                "mean_og_iou": float(np.mean(og_ious)) if og_ious else None,
                "num_sets": len(og_errs),
            })

    return results, per_set_results

if __name__ == "__main__":
    TRAIN_SETS = "sets/dev_sets.csv"
    MAPPING = "bacillales_1385_3families_geneID_to_ogID.csv"
    EMBEDDINGS = "bacillales_1385_3families_embeddings.pt"
    
    precompute_and_save_edges(
        sets_csv=TRAIN_SETS,
        mapping_csv=MAPPING,
        embeddings_path=EMBEDDINGS,
        out_dir="precomputed_edges_dev",
        max_pos=20000,
        max_neg=20000,
        pos_per_group_cap=1500,
        require_both_classes=True,
        base_seed=42
    )
    
    EDGES_DIR = "precomputed_edges_dev"
    train_set_model(
        sets_csv=TRAIN_SETS,
        mapping_csv=MAPPING,
        embeddings_path=EMBEDDINGS,
        edges_dir="precomputed_edges_dev",
        out_path="set_gnn_edge_model.pt",
        seed=42,
        epochs=15,
        lr=1e-3,
        val_frac=0.2,
        knn_k=50,
        batch_sets=1
    )
    # Build the test GeneSetDataset from your test CSV
    test_gs = GeneSetDataset(
        sets_csv="sets/test_sets.csv",
        mapping_csv=MAPPING,           
        embeddings_path=EMBEDDINGS 
    )

    test_ds = SetEdgeSupervisionDataset(
        gene_set_ds=test_gs,
        max_pos=20000,
        max_neg=20000,
        pos_per_group_cap=1500,
        base_seed=4242,
        require_both_classes=True,
        hard_neg_frac=0.7,
        hard_k=50,
        device_for_hard="mps",    
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_sets)
    model,ckpt=load_trained_model("set_gnn_edge_model_epoch015.pt","mps")
    test_loss, test_acc = eval_loss_acc(model, test_loader, device="mps", knn_k=50, thr=0.65)
    print(f"TEST (external file) | loss={test_loss:.4f}, acc={test_acc:.4f}")
    
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    device = torch.device(device)
    
    
    print("\nRunning: partition evaluation on dev sets")
    model, ckpt = load_trained_model("set_gnn_edge_model.pt", device=device)
    gene_set=GeneSetDataset(TRAIN_SETS, MAPPING, EMBEDDINGS)
    
    model.eval()
    dev_results, dev_per_set = partitions_mcl(
        model=model,
        dataset=gene_set,
        set_indices=list(range(len(gene_set))),
        knn_k=50,
        thresholds=[0.9,0.93],
        inflations=[2.0,2.2],
        device=device
    )

    dev_df = pd.DataFrame(dev_results)
    dev_per_set_df = pd.DataFrame(dev_per_set)

    print("\n=== DEV SUMMARY ===")
    print(
        dev_df.sort_values(
            by=["mean_og_error", "mean_og_iou", "exact_og_accuracy"],
            ascending=[True, False, False]
        )
    )

    dev_df.to_csv("mcl_dev_summary.csv", index=False)
    dev_per_set_df.to_csv("mcl_dev_per_set.csv", index=False)

    print("\n Test Results:")
    model, ckpt = load_trained_model("set_gnn_edge_model.pt", device=device)
    gene_set=GeneSetDataset(TRAIN_SETS, MAPPING, EMBEDDINGS)
    
    model.eval()
    t_results, t_per_set = partitions_mcl(
        model=model,
        dataset=test_gs,
        set_indices=list(range(len(gene_set))),
        knn_k=50,
        thresholds=[0.9],
        inflations=[2.2],
        device=device
    )

    t_df = pd.DataFrame(t_results)
    t_per_set_df = pd.DataFrame(t_per_set)

    print("\n=== Test SUMMARY ===")
    print(
        t_df.sort_values(
            by=["mean_og_error", "mean_og_iou", "exact_og_accuracy"],
            ascending=[True, False, False]
        )
    )

    t_df.to_csv("mcl_t_summary.csv", index=False)
    t_per_set_df.to_csv("mcl_t_per_set.csv", index=False)