from data_loader import load_data
from model import build_model

MAX_LEN = 1000
EPOCHS = 15
BATCH_SIZE = 32

X_train, X_test, y_train, y_test, vocab_size = load_data(
    "data/swissprot_binary_sampled.csv",
    max_len=MAX_LEN,
)

model = build_model(MAX_LEN, vocab_size)

model.summary()

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

model.save("models/predictor.keras")
print("Model saved.")
