"""
Build the training data of the surrogate network (real Texcoco week + 270 emulator days)
and save the train/val/test splits.

    python scripts/prepare_training_data.py     -> data/processed/  (~30 s)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.data import build_training_data
from src.utils.config import ensure_dir, load_config


# Build the data, save one CSV per split and print a summary
def main():
    cfg = load_config()
    t0 = time.time()
    df = build_training_data(cfg)
    out = ensure_dir(cfg["nn"]["data"]["output_dir"])
    for split in ("train", "val", "test"):
        df[df["split"] == split].to_csv(out / f"{split}.csv", index=False)

    summary = (df.groupby(["split", "source"])
                 .agg(days=("episode", "nunique"), rows=("episode", "size"), valid=("valid", "sum"))
                 .reset_index())
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\ncontrol styles (emulator days):")
    print(df[df["source"] == "emulator"].groupby("policy")["episode"].nunique().to_string())
    print(f"\nSaved to {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
