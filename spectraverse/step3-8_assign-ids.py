"""Step 3-8: Assign dataset-branded entry IDs.

Rewrites the primary TITLE field of every spectrum in the final MGF and the
matching TITLE column of the final CSV using a configurable prefix and
zero-padded sequence number. Designed to be a clean no-op when no prefix is
configured, so upstream Spectraverse builds keep their ``SPECTRAVERSE         N``
identifiers untouched.

Configuration (env vars):
    SPECTRAVERSE_ID_PREFIX   prefix string, e.g. "MERLIN". Unset/empty -> no-op.
    SPECTRAVERSE_ID_WIDTH    zero-padding width for the numeric suffix.
                             Default: 9 (matches MERLIN000000001 convention).

CLI usage (matches the orchestrator convention):
    python step3-8_assign-ids.py <input_csv> <input_mgf> <output_csv> <output_mgf>

The MGF carries one TITLE line per BEGIN IONS block (the primary ID; the origin
filename is a separate ORIGIN_DATASET field, see step3-7_metadata.py). This script
rewrites that first TITLE in each block.

In provenance mode (SPECTRAVERSE_ID_FROM_PROVENANCE=1, used by train_v2) the id is
the deterministic native-or-locator ORIGINAL_ID from step3-7 rather than a counter.
"""
import os
import shutil
import sys

import pandas as pd

print("Step3-8: Assigning dataset-branded entry IDs")

input_csv  = sys.argv[1]
input_mgf  = sys.argv[2]
output_csv = sys.argv[3]
output_mgf = sys.argv[4]

prefix = os.environ.get("SPECTRAVERSE_ID_PREFIX", "").strip()
width_raw = os.environ.get("SPECTRAVERSE_ID_WIDTH", "9").strip()
# Provenance mode (train_v2): the id IS the deterministic native-or-locator ORIGINAL_ID from step3-7
# (enveda/Spectraverse native TITLE, else SOURCE#INDEX), re-run-stable and collision-free — NOT the
# order-dependent counter. `prefix` then acts only as an optional namespace (left empty in v2).
from_provenance = os.environ.get("SPECTRAVERSE_ID_FROM_PROVENANCE", "").strip().lower() in ("1", "true", "yes")
try:
    width = int(width_raw)
except ValueError:
    width = 9

if not prefix and not from_provenance:
    print("[step3-8] no id scheme configured (prefix unset, provenance mode off); passing through.")
    if os.path.abspath(input_csv) != os.path.abspath(output_csv):
        shutil.copyfile(input_csv, output_csv)
    if os.path.abspath(input_mgf) != os.path.abspath(output_mgf):
        shutil.copyfile(input_mgf, output_mgf)
    sys.exit(0)

print(f"[step3-8 config] from_provenance={from_provenance}, prefix={prefix!r}, width={width}")


# --- CSV: rewrite TITLE column ---
df = pd.read_csv(input_csv, low_memory=False)
if "TITLE" not in df.columns:
    raise SystemExit("[step3-8] Expected TITLE column missing from input CSV; refusing to write.")

if from_provenance:
    if "ORIGINAL_ID" not in df.columns:
        raise SystemExit("[step3-8] SPECTRAVERSE_ID_FROM_PROVENANCE set but ORIGINAL_ID column missing "
                         "(step3-7 must run first).")
    ids = df["ORIGINAL_ID"].astype(str).tolist()
    if prefix:  # optional namespace; v2 leaves prefix empty -> native/locator ids verbatim
        ids = [f"{prefix}:{x}" for x in ids]
else:
    ids = [f"{prefix}{i + 1:0{width}d}" for i in range(len(df))]
df["TITLE"] = ids
csv_tmp = output_csv + ".tmp"
df.to_csv(csv_tmp, index=False)
os.replace(csv_tmp, output_csv)
n_csv_rows = len(df)
print(f"[step3-8] CSV: rewrote {n_csv_rows:,} TITLE values -> {output_csv}")


# --- MGF: stream line-by-line; rewrite ONLY the first TITLE per spectrum block ---
mgf_tmp = output_mgf + ".tmp"
with open(input_mgf, "r") as fin, open(mgf_tmp, "w") as fout:
    in_block = False
    title_seen_in_block = False
    spectrum_idx = 0
    for line in fin:
        if line.startswith("BEGIN IONS"):
            in_block = True
            title_seen_in_block = False
            fout.write(line)
            continue
        if line.startswith("END IONS"):
            in_block = False
            spectrum_idx += 1
            fout.write(line)
            continue
        if in_block and line.startswith("TITLE=") and not title_seen_in_block:
            new_id = ids[spectrum_idx] if spectrum_idx < len(ids) else f"{prefix}{spectrum_idx + 1:0{width}d}"
            fout.write(f"TITLE={new_id}\n")
            title_seen_in_block = True
            continue
        fout.write(line)
os.replace(mgf_tmp, output_mgf)

if spectrum_idx != n_csv_rows:
    print(
        f"[step3-8] WARNING: MGF block count ({spectrum_idx}) != CSV row count ({n_csv_rows}); "
        f"sequence numbering still respects MGF order, but inputs may be out of sync."
    )
print(f"[step3-8] MGF: rewrote primary TITLE of {spectrum_idx:,} spectra -> {output_mgf}")
