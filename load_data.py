"""Build data/knots.csv from the KnotInfo database.

Run:  python load_data.py
"""
from knotml import DATA_PATH, build_dataset

if __name__ == "__main__":
    df = build_dataset()
    print(f"Saved {len(df)} knots to {DATA_PATH.relative_to(DATA_PATH.parent.parent)}")
    print(df["crossing_number"].value_counts().sort_index().to_string())
