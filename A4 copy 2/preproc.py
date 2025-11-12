# preprocess_group_chunked.py
# Caden Anton - CIS6530 Assignment 4
# Preprocess opcode files: chunking + group column for StratifiedGroupKFold

from pathlib import Path
import pandas as pd

RAW_ROOT = Path("data_raw")   # each subfolder = APT label
OUT_DIR  = Path("data_proc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1000  # number of opcodes per chunk
MIN_TOKENS = 10

def extract_opcodes_from_text(raw: str) -> list[str]:
    """Return list of lowercase opcode mnemonics from raw disassembly text."""
    ops = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        _, rest = line.split(":", 1)
        rest = rest.strip()
        if not rest:
            continue
        op = rest.split()[0].lower().split(".")[0]
        ops.append(op)
    return ops

def chunk_list(tokens: list[str], chunk_size: int) -> list[list[str]]:
    """Split token list into fixed-size chunks."""
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

def build_dataset():
    rows = []
    for label_dir in RAW_ROOT.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for f in label_dir.glob("*.opcode"):
            raw = f.read_text(errors="ignore")
            tokens = extract_opcodes_from_text(raw)
            if len(tokens) < MIN_TOKENS:
                continue
            chunks = chunk_list(tokens, CHUNK_SIZE)
            for idx, chunk in enumerate(chunks):
                rows.append({
                    "text": " ".join(chunk),
                    "label": label,
                    "group": f.stem  # group = original file name
                })
    df = pd.DataFrame(rows)
    print(f"Total chunks: {len(df)}")
    print("Class counts:\n", df["label"].value_counts())
    df.to_csv(OUT_DIR / "grouped_chunks.csv", index=False)
    print(f"✅ Saved processed data to {OUT_DIR/'grouped_chunks.csv'}")

if __name__ == "__main__":
    build_dataset()
