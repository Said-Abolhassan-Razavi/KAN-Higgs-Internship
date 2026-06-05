"""
Run this script ONCE from the terminal to download the dataset.

    python download_data.py

It will download ~14 GB and extract the parquet file automatically.
"""
import os, urllib.request, zipfile

# ── destination ─────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
ZIP_PATH = os.path.join(DATA_DIR, "FAIR_Universe_HiggsML_data.zip")
PQ_PATH  = os.path.join(DATA_DIR, "FAIR_Universe_HiggsML_data.parquet")
ZIP_URL  = ("https://zenodo.org/api/records/15131565/files/"
            "FAIR_Universe_HiggsML_data.zip/content")

os.makedirs(DATA_DIR, exist_ok=True)
print(f"Save folder : {DATA_DIR}")

# ── already done? ───────────────────────────────────────────────────
if os.path.exists(PQ_PATH):
    gb = os.path.getsize(PQ_PATH) / 1024**3
    print(f"Parquet already present ({gb:.2f} GB). Nothing to do.")
    print(f"DATA_PATH = {PQ_PATH}")
    raise SystemExit(0)

# ── progress bar ─────────────────────────────────────────────────────
class Bar:
    def __call__(self, blocks, block_size, total):
        done = min(blocks * block_size, total)
        pct  = done / total * 100 if total > 0 else 0
        n    = int(pct / 2)
        print(f"\r  [{'#'*n}{'.'*(50-n)}] {pct:5.1f}%  "
              f"{done/1024**3:.2f}/{total/1024**3:.2f} GB",
              end="", flush=True)

# ── step 1: download zip ─────────────────────────────────────────────
if os.path.exists(ZIP_PATH):
    print(f"ZIP already downloaded ({os.path.getsize(ZIP_PATH)/1024**3:.2f} GB) — skipping.")
else:
    print("STEP 1/2  Downloading ZIP (~14 GB) …")
    print("Do NOT close this window.\n")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH, reporthook=Bar())
    print(f"\n\nDownload complete — {os.path.getsize(ZIP_PATH)/1024**3:.2f} GB")

# ── step 2: extract parquet ──────────────────────────────────────────
print("\nSTEP 2/2  Extracting parquet file …")
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    names = z.namelist()
    print(f"  Contents: {names}")
    pq_name = next((n for n in names if n.endswith(".parquet")), None)
    if pq_name is None:
        raise RuntimeError(f"No .parquet found in ZIP. Contents: {names}")
    print(f"  Extracting {pq_name} …")
    z.extract(pq_name, DATA_DIR)
    extracted = os.path.join(DATA_DIR, pq_name)
    if os.path.abspath(extracted) != os.path.abspath(PQ_PATH):
        os.rename(extracted, PQ_PATH)

print(f"\nDone!  Parquet at: {PQ_PATH}")
print(f"Size : {os.path.getsize(PQ_PATH)/1024**3:.2f} GB")

# ── optional: delete zip ─────────────────────────────────────────────
ans = input("\nDelete ZIP to free ~14 GB? (y/n): ").strip().lower()
if ans == "y":
    os.remove(ZIP_PATH)
    print("ZIP deleted.")

print(f"\nAll done. In your notebook, DATA_PATH is already set to:\n  {PQ_PATH}")
