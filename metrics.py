import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    multilabel_confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
from config import PLOT_PATH
import seaborn as sns
import os


def custom_accuracy(y_true, y_pred, threshold=0.5):
    """
    Custom evaluation function for ODIR dataset:
    1. Evaluates normal/abnormal classification accuracy.
    2. For abnormal cases, evaluates multilabel classification accuracy.

    Args:
    - y_true: Ground truth labels. Shape: (batch_size, 1 + num_classes)
              First column is 'normal/abnormal', remaining are multilabel disease labels.
    - y_pred: Model predictions. Shape: (batch_size, 1 + num_classes)
              First column is 'normal/abnormal', remaining are multilabel disease predictions.
    - threshold: Decision threshold for normal/abnormal and multilabel classification.

    Returns:
    - normal_accuracy: Accuracy of normal/abnormal classification.
    - multilabel_accuracy: Accuracy of multilabel predictions for abnormal cases.
    """
    # Split the normal/abnormal and multilabel parts
    y_true = tf.cast(y_true, tf.float32)
    normal_true = tf.cast(y_true[:, 0], tf.float32)  # Ground truth for normal/abnormal
    multilabel_true = tf.cast(y_true[:, 1:], tf.float32)  # Ground truth for diseases
    normal_pred = y_pred[:, 0]
    multilabel_pred = y_pred[:, 1:]
    # Threshold predictions
    normal_pred_binary = tf.cast(
        normal_pred > threshold, tf.float32
    )  # Binary: normal (1) or abnormal (0)
    multilabel_pred_binary = tf.cast(
        multilabel_pred > threshold, tf.float32
    )  # Binary: presence of diseases
    overall_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    # Evaluate normal/abnormal accuracy
    normal_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(normal_true, normal_pred_binary), tf.float32)
    )

    # Mask for abnormal cases (normal_true == 0)
    abnormal_mask = tf.cast(normal_true == 0, tf.float32)

    # Evaluate multilabel accuracy for abnormal cases
    # Compare predictions and ground truth for abnormal cases only
    correct_multilabel = tf.reduce_sum(
        abnormal_mask
        * tf.cast(
            tf.reduce_all(
                tf.equal(
                    multilabel_true,
                    multilabel_pred_binary,
                ),
                axis=-1,
            ),
            tf.float32,
        ),
        axis=0,
    )
    possible_labels = tf.reduce_sum(abnormal_mask)
    multilabel_accuracy = correct_multilabel / possible_labels

    overall_accuracy = tf.reduce_mean(
        tf.cast(
            tf.reduce_all(
                tf.equal(
                    y_true,
                    overall_pred_binary,
                ),
                axis=-1,
            ),
            tf.float32,
        ),
        axis=0,
    )
    batch_size, num_label = tf.shape(y_true)
    overall_label_wise_accuracy = tf.reduce_sum(
        tf.cast(
            tf.equal(
                y_true,
                overall_pred_binary,
            ),
            dtype=tf.float32,
        ),
    ) / tf.cast(batch_size * num_label, dtype=tf.float32)

    return (
        normal_accuracy.numpy(),
        multilabel_accuracy.numpy(),
        overall_accuracy.numpy(),
        overall_label_wise_accuracy.numpy(),
    )


def f_1_score(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, average="macro")
    return f1


def auc_roc(y_true, y_pred):
    roc_auc = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")
    return roc_auc


def plot_confusion_matrix(
    y_true,
    y_pred,
    threshold=0.5,
    epoch=0,
    model_name="",
):
    y_pred = (y_pred >= threshold).astype(int)
    conf_matrices = multilabel_confusion_matrix(y_true, y_pred)
    # Plot each confusion matrix as a heatmap
    num_labels = conf_matrices.shape[0]
    fig, axes = plt.subplots(1, num_labels, figsize=(20, 8))

    for i, matrix in enumerate(conf_matrices):
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i])
        axes[i].set_title(f"Label {i}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    # Adjust layout and save as an image
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"confusion_matrix_{epoch}_{model_name}.png"))
