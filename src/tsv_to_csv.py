import pandas as pd

TSV_PATH = "data/uniprot_sprot.tsv"
OUTPUT_CSV = "data/swissprot_binary.csv"

MAX_LEN = 1000
MIN_LEN = 50


def main():
    print("Loading TSV...")
    df = pd.read_csv(TSV_PATH, sep="\t")

    print("Columns found:", df.columns.tolist())

    # Filter short sequences
    df = df[df["Sequence"].str.len() >= MIN_LEN]

    # Truncate long sequences
    df["Sequence"] = df["Sequence"].str.slice(0, MAX_LEN)

    # Binary label: enzyme = 1 if EC number exists
    df["label"] = df["EC number"].notna().astype(int)

    final_df = df[["Sequence", "label"]]
    final_df.columns = ["sequence", "label"]

    print("Label distribution:")
    print(final_df["label"].value_counts())

    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved dataset to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
