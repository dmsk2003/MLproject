import requests
import csv
import time
import random
from typing import List, Optional, Dict

# CONFIG
API_BASE = "https://data.orthodb.org/v12"

PARENT_LEVEL = "1385"  # Bacillales

FAMILIES = {
    "186817": "Bacillaceae",
    "186822": "Paenibacillaceae",
    "186823": "Alicyclobacillaceae"
}

OUTPUT_FILE = "bacillales_1385_3families_geneID_to_ogID.csv"

# Paging for /search (OG ids)
TAKE = 5000  # OrthoDB allows up to 10000 per docs; 5000 is a safe chunk
SAMPLE_OGS = True
MAX_OGS_TO_PROCESS = 15000

# Rate limiting
SLEEP_BETWEEN_SEARCH_CALLS = 0.15
SLEEP_BETWEEN_TAB_CALLS = 0.25 

RANDOM_SEED = 42


def fetch_json_with_retry(url, params=None, retries=4, sleep_time=1.0) -> Optional[Dict]:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=45)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException as e:
            print(f"[json] attempt {attempt+1} failed: {e}")
        time.sleep(sleep_time)
    return None

def fetch_text_with_retry(url, params=None, retries=4, sleep_time=1.0) -> Optional[str]:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=45)
            if r.status_code == 200:
                return r.text
        except requests.RequestException as e:
            print(f"[text] attempt {attempt+1} failed: {e}")
        time.sleep(sleep_time)
    return None

# OrthoDB: fetch OG ids at a level
def fetch_all_og_ids_at_level(level: str) -> List[str]:
    search_url = f"{API_BASE}/search"
    skip = 0
    all_ogs: List[str] = []
    total = None

    while True:
        j = fetch_json_with_retry(search_url, params={"level": level, "take": TAKE, "skip": skip})
        if j is None:
            break

        if total is None:
            total = j.get("count")
            print(f"Total OG clusters at level={level}: {total}")

        batch = j.get("data", [])
        all_ogs.extend(batch)

        print(f"Fetched {len(all_ogs)} OG IDs so far... (skip={skip})")

        if len(batch) < TAKE:
            break

        skip += TAKE
        time.sleep(SLEEP_BETWEEN_SEARCH_CALLS)

    return all_ogs


# OrthoDB: parse /tab TSV
# Columns per docs:
# pub_og_id, og_name, level_taxid, organism_taxid, organism_name, pub_gene_id, description
# :contentReference[oaicite:1]{index=1}
def parse_tab_tsv(tsv_text: str):
    """
    Yields dict rows with keys from the header line.
    """
    lines = [ln for ln in tsv_text.splitlines() if ln.strip()]
    if not lines:
        return

    header = lines[0].split("\t")
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) != len(header):
            continue
        yield dict(zip(header, parts))

# =======================
# MAIN
# =======================
def main():
    random.seed(RANDOM_SEED)

    # Get Bacillales-level OG ids 
    og_ids = fetch_all_og_ids_at_level(PARENT_LEVEL)
    if not og_ids:
        raise RuntimeError("No OG IDs fetched. Check API / connectivity.")

    # Sample for compute safety
    if SAMPLE_OGS and len(og_ids) > MAX_OGS_TO_PROCESS:
        random.shuffle(og_ids)
        og_ids = og_ids[:MAX_OGS_TO_PROCESS]
        print(f"Sampling enabled: processing {len(og_ids)} OGs (out of all).")
    else:
        print(f"Processing ALL fetched OGs: {len(og_ids)}")

    tab_url = f"{API_BASE}/tab"

    seen_gene_ids = set()
    kept_rows = 0
    kept_by_family = {fid: 0 for fid in FAMILIES.keys()}

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gene_id", "og_id"])

        for i, og_id in enumerate(og_ids, start=1):
            if i % 25 == 0:
                print(f"[{i}/{len(og_ids)}] OG={og_id} | kept_rows={kept_rows}")

            for family_taxid in FAMILIES.keys():
                tsv = fetch_text_with_retry(tab_url, params={"id": og_id, "species": family_taxid})
                time.sleep(SLEEP_BETWEEN_TAB_CALLS)

                if not tsv:
                    continue

                for row in parse_tab_tsv(tsv):
                    gene_id = row.get("pub_gene_id")
                    if not gene_id or gene_id in seen_gene_ids:
                        continue

                    writer.writerow([gene_id, og_id])
                    seen_gene_ids.add(gene_id)
                    kept_rows += 1

    print("\nDone!")
    print(f"Wrote {kept_rows} rows to {OUTPUT_FILE}")
    print(f"Kept by family: {kept_by_family}")

if __name__ == "__main__":
    main()