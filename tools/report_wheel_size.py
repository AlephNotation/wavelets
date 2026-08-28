"""Measure compressed and installed sizes of one wheel."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

NATIVE_SUFFIXES = (".so", ".pyd", ".dylib")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel_directory", type=Path)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    wheels = sorted(args.wheel_directory.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel, found {len(wheels)}: {wheels}")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        files = [entry for entry in archive.infolist() if not entry.is_dir()]
        native_files = [
            entry for entry in files if entry.filename.endswith(NATIVE_SUFFIXES)
        ]
    if len(native_files) != 1:
        raise SystemExit(
            f"expected one native extension, found {len(native_files)}: "
            f"{[entry.filename for entry in native_files]}"
        )

    result = {
        "platform": args.platform,
        "python": args.python,
        "wheel": wheel.name,
        "wheel_bytes": wheel.stat().st_size,
        "unpacked_bytes": sum(entry.file_size for entry in files),
        "native_extension": native_files[0].filename,
        "native_extension_bytes": native_files[0].file_size,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(
                "| Platform | Python | Wheel | Compressed | Unpacked | Native |\n"
                "|---|---:|---|---:|---:|---:|\n"
                f"| {result['platform']} | {result['python']} | "
                f"`{result['wheel']}` | {result['wheel_bytes']:,} B | "
                f"{result['unpacked_bytes']:,} B | "
                f"{result['native_extension_bytes']:,} B |\n"
            )


if __name__ == "__main__":
    main()
