"""Loads data/processed/diabetic_data_features.csv into a local SQLite db
(data/processed/carerisk.db, gitignored) so the queries in this folder are runnable."""
import sqlite3
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("data/processed/diabetic_data_features.csv")
    con = sqlite3.connect("data/processed/carerisk.db")
    df.to_sql("encounters", con, if_exists="replace", index=False)
    con.close()
    print(f"Loaded {len(df)} rows into data/processed/carerisk.db (table: encounters)")
