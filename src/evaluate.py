"""
evaluate.py — Model evaluation & metrics.
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.config import cfg
from src.data_preprocessing import create_test_generator
from src.model import load_trained_model
from src.utils import plot_confusion_matrix


def evaluate(model_path: str = None):
    """Evaluate the trained model on the test set."""
    print("=" * 60)
    print("  Real-Time Face Analysis — Evaluation")
    print("=" * 60)

    model = load_trained_model(model_path)
    test_gen = create_test_generator()
    print(f"  Test samples: {test_gen.samples}")

    loss, accuracy = model.evaluate(test_gen, verbose=1)
    print(f"\n  Test Loss:     {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f}")

    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, cfg.paths.confusion_matrix_plot)
    print(f"Confusion matrix saved to: {cfg.paths.confusion_matrix_plot}")

    return {"loss": loss, "accuracy": accuracy, "confusion_matrix": cm}
