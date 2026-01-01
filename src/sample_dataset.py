import pandas as pd

INPUT_CSV = "data/swissprot_binary.csv"
OUTPUT_CSV = "data/swissprot_binary_sampled.csv"

SAMPLES_PER_CLASS = 50000  # total = 100k


def main():
    df = pd.read_csv(INPUT_CSV)

    df_0 = df[df["label"] == 0].sample(SAMPLES_PER_CLASS, random_state=42)
    df_1 = df[df["label"] == 1].sample(SAMPLES_PER_CLASS, random_state=42)

    sampled = pd.concat([df_0, df_1]).sample(frac=1, random_state=42)

    print(sampled["label"].value_counts())

    sampled.to_csv(OUTPUT_CSV, index=False)
    print("Saved:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
