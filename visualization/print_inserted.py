#!/usr/bin/env python3
import sys, glob

def last_inserted_count(csv_path: str) -> int:
    last = 0
    with open(csv_path) as f:
        next(f, None)  # skip header
        for line in f:
            if not line.strip():
                continue
            last = int(line.split(",")[0])
    return last

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <run_dir> [<run_dir> ...]")
        sys.exit(1)

    for run_dir in sys.argv[1:]:
        logs = sorted(glob.glob(f"{run_dir}/*/fitness_log.csv"))
        if not logs:
            print(f"{run_dir}: no fitness_log.csv files found")
            continue

        counts = [last_inserted_count(p) for p in logs]
        print(f"{run_dir}: per-log final inserted_genomes = {counts}, max = {max(counts)}")

