import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
CHAR_TO_INT = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
VOCAB_SIZE = len(AMINO_ACIDS) + 1


def encode_sequence(seq):
    return [CHAR_TO_INT.get(aa, 0) for aa in seq]


def load_data(
    csv_path,
    max_len=1000,
    test_size=0.2,
):
    df = pd.read_csv(csv_path)

    sequences = df["sequence"].tolist()
    labels = df["label"].values

    encoded = [encode_sequence(seq) for seq in sequences]
    padded = pad_sequences(encoded, maxlen=max_len, padding="post", truncating="post")

    X_train, X_test, y_train, y_test = train_test_split(
        padded, labels, test_size=test_size, random_state=42, stratify=labels
    )

    return X_train, X_test, y_train, y_test, VOCAB_SIZE
