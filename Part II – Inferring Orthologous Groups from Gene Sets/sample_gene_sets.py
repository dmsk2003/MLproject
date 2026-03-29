import os
import random
from collections import defaultdict

import pandas as pd
import torch


OG_FILE = "bacillales_1385_3families_geneID_to_ogID.csv"
EMBEDDINGS_FILE = "bacillales_1385_3families_embeddings.pt"

OUT_DIR = "semiunsup_random_sets_onefile"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
DEV_FRACTION = 0.7  # split by OG

# Desired set sizes
SET_SIZES =[1000,2000,2500,3000,4000]
SETS_PER_SIZE = 100

# Cap how many genes per OG are considered "usable" to avoid huge OG dominance
CAP_PER_OG = 250

# OG eligibility: OG must have at least this many usable genes
MIN_OG_SIZE = 2

# Variable contribution per OG inside each set
MIN_GENES_PER_OG = 1
MAX_GENES_PER_OG = 350

# "uniform" / "loguniform" / "geometric"
PER_OG_DISTRIBUTION = "loguniform"

# Safety caps
MAX_OGS_PER_SET = 500
RESTARTS_PER_SET = 80


def load_og_df(path):
    df = pd.read_csv(path)
    df["gene_id"] = df["gene_id"].astype(str)
    df["og_id"] = df["og_id"].astype(str)
    return df


def load_embeddings_keys(path):
    emb = torch.load(path, map_location="cpu", weights_only=False)
    return set(str(k) for k in emb.keys())


def build_og_to_genes(df_og)]:
    og_to_genes = defaultdict(list)
    for r in df_og.itertuples(index=False):
        og_to_genes[r.og_id].append(r.gene_id)
    return og_to_genes


def draw_k(rng, min_k, max_k, distribution) :
    if max_k <= 0:
        return 0
    min_k = max(1, min_k)
    max_k = max(min_k, max_k)

    if distribution == "uniform":
        return rng.randint(min_k, max_k)

    if distribution == "loguniform":
        import math
        a = math.log(min_k)
        b = math.log(max_k)
        return int(round(math.exp(rng.uniform(a, b))))

    if distribution == "geometric":
        p = 0.35
        k = 1
        while k < max_k and rng.random() > p:
            k += 1
        return max(min_k, min(k, max_k))

    raise ValueError(f"Unknown distribution: {distribution}")


def prepare_eligible_ogs(
    og_to_genes,
    og_pool,
    rng,
    cap_per_og,
    min_og_size,
):
    """
    Returns:
      eligible_ogs: list[str]
      usable_pool: dict[og_id] -> list[gene_id] (capped)
    """
    eligible_ogs = []
    usable_pool = {}

    for og in og_pool:
        genes = og_to_genes.get(og, [])
        if not genes:
            continue

        # cap the pool for this OG
        if len(genes) > cap_per_og:
            pool = rng.sample(genes, cap_per_og)
        else:
            pool = list(genes)

        if len(pool) >= min_og_size:
            eligible_ogs.append(og)
            usable_pool[og] = pool

    if not eligible_ogs:
        raise ValueError(
            "No eligible OGs found. Try lowering MIN_OG_SIZE or increasing CAP_PER_OG."
        )

    return eligible_ogs, usable_pool


def generate_variable_og_sets(
    og_to_genes,
    og_pool,
    target_sizes,
    sets_per_size,
    seed,
    cap_per_og,
    min_og_size,
    min_genes_per_og,
    max_genes_per_og,
    per_og_distribution,
    max_ogs_per_set,
    restarts_per_set,
):
    """
    Create sets with VARIABLE genes-per-OG.
    Output CSV column MUST be named 'gene_ids' (comma-separated gene IDs).
    """
    rng = random.Random(seed)

    eligible_ogs, usable_pool = prepare_eligible_ogs(
        og_to_genes=og_to_genes,
        og_pool=og_pool,
        rng=rng,
        cap_per_og=cap_per_og,
        min_og_size=min_og_size,
    )

    lines = []
    sizes = []

    for size in target_sizes:
        for _ in range(sets_per_size):
            ok = False

            for _attempt in range(restarts_per_set):
                remaining = size
                chosen_genes = []
                used_gene_ids = set()
                used_ogs = set()

                og_order = eligible_ogs[:]
                rng.shuffle(og_order)

                for og in og_order:
                    if remaining == 0:
                        ok = True
                        break
                    if len(used_ogs) >= max_ogs_per_set:
                        break

                    pool = usable_pool[og]
                    # avoid duplicates across the set
                    available = [g for g in pool if g not in used_gene_ids]
                    if not available:
                        continue

                    max_take = min(len(available), remaining, max_genes_per_og)
                    if max_take <= 0:
                        continue

                    min_take = min(min_genes_per_og, max_take)
                    take = draw_k(rng, min_take, max_take, per_og_distribution)

                    picked = rng.sample(available, take)
                    chosen_genes.extend(picked)
                    used_gene_ids.update(picked)
                    used_ogs.add(og)
                    remaining -= take

                if remaining == 0:
                    rng.shuffle(chosen_genes)
                    lines.append(",".join(chosen_genes))
                    sizes.append(size)
                    ok = True
                    break

            if not ok:
                raise RuntimeError(
                    f"Failed to generate a full set of size={size}. "
                    f"Try increasing CAP_PER_OG, lowering MIN_OG_SIZE, "
                    f"increasing MAX_OGS_PER_SET, or reducing MAX_GENES_PER_OG."
                )

    return lines, sizes


def main():
    print("Loading OG mapping...")
    df_og = load_og_df(OG_FILE)

    print("Loading embeddings keys...")
    emb_genes = load_embeddings_keys(EMBEDDINGS_FILE)

    print("Filtering genes without embeddings...")
    df_og = df_og[df_og["gene_id"].isin(emb_genes)].copy()

    print("Building OG -> genes...")
    og_to_genes = build_og_to_genes(df_og)

    print("Splitting OGs into dev/test...")
    ogs = list(og_to_genes.keys())
    rng_split = random.Random(SEED)
    rng_split.shuffle(ogs)

    cut = int(len(ogs) * DEV_FRACTION)
    dev_ogs = ogs[:cut]
    test_ogs = ogs[cut:]

    split_path = os.path.join(OUT_DIR, "split_ogs.csv")
    split_rows = [{"og_id": og, "split": "dev"} for og in dev_ogs] + \
                 [{"og_id": og, "split": "test"} for og in test_ogs]
    pd.DataFrame(split_rows).to_csv(split_path, index=False)
    print(f"Saved split file: {split_path}")

    print("Generating DEV sets...")
    dev_lines, dev_sizes = generate_variable_og_sets(
        og_to_genes=og_to_genes,
        og_pool=dev_ogs,
        target_sizes=SET_SIZES,
        sets_per_size=SETS_PER_SIZE,
        seed=SEED + 10,
        cap_per_og=CAP_PER_OG,
        min_og_size=MIN_OG_SIZE,
        min_genes_per_og=MIN_GENES_PER_OG,
        max_genes_per_og=MAX_GENES_PER_OG,
        per_og_distribution=PER_OG_DISTRIBUTION,
        max_ogs_per_set=MAX_OGS_PER_SET,
        restarts_per_set=RESTARTS_PER_SET,
    )

    print("Generating TEST sets...")
    test_lines, test_sizes = generate_variable_og_sets(
        og_to_genes=og_to_genes,
        og_pool=test_ogs,
        target_sizes=SET_SIZES,
        sets_per_size=SETS_PER_SIZE,
        seed=SEED + 20,
        cap_per_og=CAP_PER_OG,
        min_og_size=MIN_OG_SIZE,
        min_genes_per_og=MIN_GENES_PER_OG,
        max_genes_per_og=MAX_GENES_PER_OG,
        per_og_distribution=PER_OG_DISTRIBUTION,
        max_ogs_per_set=MAX_OGS_PER_SET,
        restarts_per_set=RESTARTS_PER_SET,
    )

    # IMPORTANT: must contain column 'gene_ids'
    dev_path = os.path.join(OUT_DIR, "dev_sets.csv")
    test_path = os.path.join(OUT_DIR, "test_sets.csv")

    dev_df = pd.DataFrame({"gene_ids": dev_lines, "size": dev_sizes})
    test_df = pd.DataFrame({"gene_ids": test_lines, "size": test_sizes})

    dev_df.to_csv(dev_path, index=False)
    test_df.to_csv(test_path, index=False)

    # sanity checks (fail fast if something went wrong)
    assert "gene_ids" in pd.read_csv(dev_path).columns, "DEV file missing 'gene_ids' column"
    assert "gene_ids" in pd.read_csv(test_path).columns, "TEST file missing 'gene_ids' column"

    print("Done.")
    print(f"DEV sets file:  {dev_path}  (rows={len(dev_lines)})")
    print(f"TEST sets file: {test_path} (rows={len(test_lines)})")


if __name__ == "__main__":
    main()