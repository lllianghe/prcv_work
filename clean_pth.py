#!/usr/bin/env python3

# 使用示例
# python3 /SSD_Data01/lhe/myfile/multimodel_exp/base/prcv_work/clean_pth.py --verbose --path /SSD_Data01/lhe/myfile/multimodel_exp/base/prcv_work/logs

import argparse
import os
import sys
from typing import List, Tuple


def find_pth_files(root_dir: str) -> List[str]:
    """Return a list of absolute paths to *.pth files under root_dir (recursive)."""
    matched_files: List[str] = []
    for current_dir, _subdirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith(".pth"):
                matched_files.append(os.path.join(current_dir, filename))
    return matched_files


def remove_files(file_paths: List[str], dry_run: bool = False) -> Tuple[int, int]:
    """
    Remove files in file_paths.
    Returns (num_deleted, num_errors).
    """
    num_deleted = 0
    num_errors = 0
    for path in file_paths:
        if dry_run:
            continue
        try:
            os.remove(path)
            num_deleted += 1
        except Exception:
            num_errors += 1
    return num_deleted, num_errors


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Recursively delete all .pth files under a directory (default: ./logs).",
    )
    parser.add_argument(
        "--path",
        default="logs",
        help="Target directory to scan recursively (default: logs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files to be deleted without removing them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each matched file.",
    )

    args = parser.parse_args(argv)

    target_dir = os.path.abspath(args.path)
    if not os.path.exists(target_dir):
        print(f"[ERROR] Path does not exist: {target_dir}")
        return 1
    if not os.path.isdir(target_dir):
        print(f"[ERROR] Path is not a directory: {target_dir}")
        return 1

    matched = find_pth_files(target_dir)
    if args.verbose or args.dry_run:
        for p in matched:
            print(p)

    if args.dry_run:
        print(f"[DRY-RUN] {len(matched)} .pth file(s) would be deleted under: {target_dir}")
        return 0

    deleted, errors = remove_files(matched, dry_run=False)
    print(f"[DONE] Deleted: {deleted}, Errors: {errors}, Scanned dir: {target_dir}")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


