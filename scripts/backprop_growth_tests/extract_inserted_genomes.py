import os
import csv

BASE_DIR = "."  # directory where the script is executed
OUTPUT_CSV = "output.csv"

def get_last_field0_from_csv(path):
    """Return fields[0] from the last line of a CSV file."""
    with open(path, "rb") as f:
        try:
            # Go to the byte before EOF
            f.seek(-2, os.SEEK_END)
            # Walk backwards until we hit a newline
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            # File might be only one line long; go to start
            f.seek(0)
        last_line = f.readline().decode().strip()

    fields = last_line.split(",")
    return fields[0] if fields else None


rows = []

for dirpath, dirnames, filenames in os.walk(BASE_DIR):
    if "fitness_log.csv" in filenames:
        fitness_path = os.path.join(dirpath, "fitness_log.csv")

        # dirpath is something like: ./Strategy/Experimental_Setup/Run_number
        rel_path = os.path.relpath(dirpath, BASE_DIR)
        parts = rel_path.split(os.sep)

        # Expect at least: Strategy / Experimental_Setup / Run_number
        if len(parts) < 3:
            # Skip unexpected structure
            continue

        strategy, experimental_setup, run_number = parts[-3:]

        desired_value = get_last_field0_from_csv(fitness_path)

        rows.append([
            strategy,
            experimental_setup,
            run_number,
            desired_value,
        ])

# Write the result CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Strategy", "Experimental_Setup", "Run_number", "Inserted Genomes"])
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")

