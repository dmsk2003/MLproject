import requests
import pandas as pd
import time
import json
from typing import Optional, Dict, Set

API_BASE = "https://data.orthodb.org/v12"
FASTA_URL = f"{API_BASE}/fasta"

MAPPING_FILE = "bacillales_1385_3families_geneID_to_ogID.csv"  # output
OUTPUT_SEQ_FILE = "bacillales_1385_3families_sequences.csv"

SLEEP_BETWEEN_FASTA_CALLS = 1.05 
SAVE_EVERY_OGS = 50               

def fetch_text_with_retry(url, params=None, retries=4, sleep_time=1.0) :
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.text
        except requests.RequestException:
            pass
        time.sleep(sleep_time)
    return None

def parse_header_json(header_line: str) :
    """
    Header example:
    >1280672_0:00116d {"pub_og_id":"...at1385",...,"pub_gene_id":"G602_RS30655",...}
    """
    try:
        parts = header_line[1:].split(" ", 1)  # remove '>' then split token + json
        if len(parts) < 2:
            return None
        return json.loads(parts[1])
    except Exception:
        return None

def main():
    # 1) Load mapping
    df = pd.read_csv(MAPPING_FILE, dtype=str)
    df = df.dropna()
    df["gene_id"] = df["gene_id"].astype(str)
    df["og_id"] = df["og_id"].astype(str)

    genes_needed: Set[str] = set(df["gene_id"].tolist())
    og_to_genes = df.groupby("og_id")["gene_id"].apply(set).to_dict()

    print(f"Need sequences for {len(genes_needed)} genes from {len(og_to_genes)} OGs")

    
    try:
        existing = pd.read_csv(OUTPUT_SEQ_FILE, dtype=str)
        done = set(existing["gene_id"].astype(str).tolist())
        print(f"Resuming: {len(done)} genes already in {OUTPUT_SEQ_FILE}")
    except Exception:
        existing = None
        done = set()

    rows = [] if existing is None else existing.values.tolist()
    genes_remaining = genes_needed - done
    print(f"Genes remaining to fetch: {len(genes_remaining)}")

    # Iterate OGs and fetch FASTA
    og_ids = list(og_to_genes.keys())

    for i, og_id in enumerate(og_ids, start=1):
        # skip OG if all its genes already done
        if og_to_genes[og_id].issubset(done):
            continue

        if i % 25 == 0:
            print(f"[{i}/{len(og_ids)}] OG={og_id} | sequences_collected={len(done)}")

        fasta_text = fetch_text_with_retry(FASTA_URL, params={"id": og_id})
        if not fasta_text:
            time.sleep(SLEEP_BETWEEN_FASTA_CALLS)
            continue

        current_gene = None
        current_seq_parts = []

        def flush_current():
            nonlocal current_gene, current_seq_parts, done, rows
            if current_gene and current_gene in genes_needed and current_gene not in done:
                seq = "".join(current_seq_parts)
                if seq:  # only if non-empty
                    rows.append([current_gene, seq])
                    done.add(current_gene)

        for line in fasta_text.splitlines():
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # save previous
                flush_current()

                meta = parse_header_json(line)
                current_gene = meta.get("pub_gene_id") if meta else None
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

        # last record
        flush_current()

        # checkpoint write
        if (len(done) % 500 == 0) or (i % SAVE_EVERY_OGS == 0):
            pd.DataFrame(rows, columns=["gene_id", "sequence"]).to_csv(OUTPUT_SEQ_FILE, index=False)

        time.sleep(SLEEP_BETWEEN_FASTA_CALLS)

    pd.DataFrame(rows, columns=["gene_id", "sequence"]).to_csv(OUTPUT_SEQ_FILE, index=False)
    print(f"Done! Wrote {len(done)} sequences to {OUTPUT_SEQ_FILE}")

if __name__ == "__main__":
    main()
