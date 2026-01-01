import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_loader import encode_sequence

MAX_LEN = 1000

model = load_model("models/predictor.h5")


def predict(sequence):
    encoded = encode_sequence(sequence)
    padded = pad_sequences([encoded], maxlen=MAX_LEN, padding="post")
    prob = model.predict(padded)[0][0]
    return prob


if __name__ == "__main__":
    seq = "MKWVTFISLLFLFSSAYSRGVFRR"
    print("Predicted probability:", predict(seq))
