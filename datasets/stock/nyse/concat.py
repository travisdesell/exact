import psycopg2
from datetime import datetime
import os
import csv

csv_directory = "nyse/csvs"

db_params = {"host": "localhost", "database": "NYSE", "user": "rohaan", "password": ""}
files = 0
records = 0

for file in sorted(os.listdir(csv_directory)):
    if ".csv" not in file:
        continue
    files += 1
    file_path = os.path.join(csv_directory, file)
    file_name = file.split(".")[0]
    print(f"\tProcessing {file}")

    # Connect to the PostgreSQL database
    with psycopg2.connect(**db_params) as conn:
        cursor = conn.cursor()
        with open(file_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            # skip header
            next(csv_reader)
            csv_records = 0

            for index, row in enumerate(csv_reader):
                csv_records += 1
                date_object = datetime.strptime(row[0], "%d-%m-%Y")
                formatted_date = date_object.strftime("%Y-%m-%d")
                sql_statement = f"INSERT INTO prices (stock, date, low, open, volume, high, close, adjusted_close) VALUES ('{file_name}', '{formatted_date}', {row[1]}, {row[2]}, {row[3]}, {row[4]}, {row[5]}, {row[6]})"
                try:
                    cursor.execute(sql_statement)
                    conn.commit()
                    records += 1
                except Exception as e:
                    print(f"{file_name} \t\t**generated error : {e}**")

            print(f"\t\tRecords Added: {csv_records}")
print(f"{records} added from {files} csvs")
