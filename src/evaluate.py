import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
)
from tensorflow.keras.models import load_model
from data_loader import load_data

MAX_LEN = 1000
MODEL_PATH = "models/predictor.h5"
DATA_PATH = "data/swissprot_binary_sampled.csv"


def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test, vocab_size = load_data(DATA_PATH, max_len=MAX_LEN)

    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Predicting...")
    y_probs = model.predict(X_test).ravel()
    y_pred = (y_probs >= 0.5).astype(int)

    # confusion martix
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Enzyme", "Enzyme"],
    )

    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix")

    plt.savefig("figures/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved confusion_matrix.png")

    # roc curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    plt.savefig("figures/roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved roc_curve.png")

    # clasification report
    print("\nClassification Report:\n")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Non-Enzyme", "Enzyme"],
        )
    )


if __name__ == "__main__":
    main()
