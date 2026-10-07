"""Load the raw HC3 dataset and perform basic row-level cleaning."""

import pandas as pd


def load_raw_data():
    """Read Dataset/raw_dataset.csv, drop the 'index' column and duplicate rows."""
    df = pd.read_csv("Dataset/raw_dataset.csv")
    df = df.drop(columns=["index"])
    df = df.drop_duplicates()
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print("Shape:", df.shape)
    print("\nNull counts:")
    print(df.isnull().sum())
    print("\nSource counts:")
    print(df["source"].value_counts())
