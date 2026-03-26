"""
Fix parquet files where shape=[1] fields are stored as list<T> instead of scalar T.

LeRobot's get_hf_features_from_features() treats shape=[1] as Value(dtype),
but RoboCOIN's parquet stores them as list<element: T>. This causes:
    TypeError: Couldn't cast array of type list<element: int32> to int32

This script converts list<T> columns with shape=[1] in info.json to scalar T
in the parquet files, making them compatible with LeRobot's dataset loader.

Usage:
    python fix_parquet_list_scalars.py --root <dataset_root>

Example:
    python fix_parquet_list_scalars.py \
        --root robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def get_shape1_features(info_path: Path) -> list[str]:
    """Get feature names that have shape=[1] and are not video/image."""
    with open(info_path) as f:
        info = json.load(f)

    shape1_keys = []
    for key, ft in info["features"].items():
        if ft["dtype"] in ("video", "image"):
            continue
        if ft.get("shape") == [1]:
            shape1_keys.append(key)

    return shape1_keys


def fix_parquet_file(parquet_path: Path, shape1_keys: list[str]) -> bool:
    """Fix list<T> columns to scalar T for shape=[1] features.

    Returns True if any column was fixed.
    """
    table = pq.read_table(parquet_path)
    fixed = False

    for key in shape1_keys:
        if key not in table.column_names:
            continue

        col = table.column(key)
        # Check if it's a list type that needs flattening
        if not pa.types.is_list(col.type) and not pa.types.is_large_list(col.type):
            continue

        # Extract first element from each list to make it scalar
        # e.g. list<int32> [[1], [2], [3]] -> int32 [1, 2, 3]
        flattened = col.combine_chunks()
        values = flattened.values  # the underlying flat array
        offsets = flattened.offsets.to_pylist()

        scalar_values = []
        for i in range(len(flattened)):
            start = offsets[i]
            end = offsets[i + 1]
            if end - start == 1:
                scalar_values.append(values[start].as_py())
            elif end - start == 0:
                scalar_values.append(None)
            else:
                # Multi-element list in a shape=[1] field - unexpected
                raise ValueError(
                    f"Column '{key}' has shape=[1] in info.json but row {i} "
                    f"has {end - start} elements"
                )

        new_col = pa.array(scalar_values, type=col.type.value_type)
        col_idx = table.column_names.index(key)
        table = table.set_column(col_idx, key, new_col)
        fixed = True
        print(f"  Fixed: {key} (list<{col.type.value_type}> -> {col.type.value_type})")

    if fixed:
        pq.write_table(table, parquet_path)

    return fixed


def main():
    parser = argparse.ArgumentParser(description="Fix list<T> shape=[1] columns in parquet")
    parser.add_argument("--root", required=True, help="Dataset root directory")
    args = parser.parse_args()

    root = Path(args.root)
    info_path = root / "meta" / "info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found at {info_path}")

    shape1_keys = get_shape1_features(info_path)
    if not shape1_keys:
        print("No shape=[1] features found. Nothing to fix.")
        return

    print(f"shape=[1] features to check: {shape1_keys}")

    # Find all parquet files in data/
    data_dir = root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet file(s)")

    fixed_count = 0
    for pf in parquet_files:
        print(f"Processing: {pf}")
        if fix_parquet_file(pf, shape1_keys):
            fixed_count += 1

    print(f"\nDone. Fixed {fixed_count}/{len(parquet_files)} file(s).")


if __name__ == "__main__":
    main()
