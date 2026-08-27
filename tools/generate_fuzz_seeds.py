#!/usr/bin/env python3
"""Generate small, named seed corpora for every cargo-fuzz target."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

ROOT = Path("fuzz/corpus")


def samples(values: list[int]) -> bytes:
    return b"".join(struct.pack("<h", value) for value in values)


SEEDS = {
    "dwt_roundtrip": {
        "seed-empty.bin": b"",
        "seed-haar-symmetric-one.bin": bytes([0, 2]) + samples([1]),
        "seed-db38-antireflect-odd.bin": bytes([37, 7]) + samples(list(range(-8, 9))),
        "seed-db2-periodization-f32.bin": bytes([1, 0x80 | 8])
        + samples([1, -2, 3, -4, 5, -6, 7, -8]),
        "seed-sym20-smooth-even.bin": bytes([56, 5]) + samples(list(range(-16, 16))),
        "seed-coif17-antireflect-odd.bin": bytes([73, 7])
        + samples(list(range(-16, 17))),
    },
    "wavedec_roundtrip": {
        "seed-empty.bin": b"",
        "seed-haar-max.bin": bytes([0, 2, 0x80]) + samples(list(range(16))),
        "seed-db4-exact.bin": bytes([3, 8, 2]) + samples(list(range(-32, 32))),
        "seed-invalid-level.bin": bytes([37, 2, 1]) + samples([1, 2, 3, 4]),
        "seed-sym20-max.bin": bytes([56, 8, 0x80]) + samples(list(range(-32, 32))),
        "seed-coif17-max.bin": bytes([73, 2, 0x80]) + samples(list(range(-48, 48))),
    },
    "custom_filter_bank": {
        "seed-valid.bin": bytes([0, 2, 16])
        + struct.pack("<QQ", 0x3FE6A09E667F3BCD, 0x3FE6A09E667F3BCD),
        "seed-empty.bin": bytes([1, 0, 8]),
        "seed-odd.bin": bytes([2, 5, 7]) + struct.pack("<Q", 0x3FF0000000000000),
        "seed-nonfinite.bin": bytes([4, 8, 9]) + struct.pack("<Q", 0x3FF0000000000000),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for target, seeds in SEEDS.items():
        directory = ROOT / target
        directory.mkdir(parents=True, exist_ok=True)
        for name, contents in seeds.items():
            path = directory / name
            if args.check:
                if not path.exists() or path.read_bytes() != contents:
                    raise SystemExit(f"{path} is stale; regenerate fuzz seeds")
            else:
                path.write_bytes(contents)


if __name__ == "__main__":
    main()
