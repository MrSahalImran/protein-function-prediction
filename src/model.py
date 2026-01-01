import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam


def build_model(max_len, vocab_size):
    model = Sequential(name="Protein_Function_Predictor")

    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=128,
            input_length=max_len,
        )
    )

    model.add(Conv1D(128, 7, activation="relu", padding="same"))
    model.add(MaxPooling1D(2))

    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(GlobalMaxPooling1D())

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
