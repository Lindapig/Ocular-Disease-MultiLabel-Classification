import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def custom_loss(y_true, y_pred, rho=1):
    """
    Custom loss function combining normal-abnormal classification and
    multilabel classification for abnormal images.

    Args:
    - y_true: Ground truth tensor of shape (batch_size, 1 + num_classes).
              First column is 'normal', remaining are multilabels.
    - y_pred: Predicted tensor of shape (batch_size, 1 + num_classes).

    Returns:
    - Combined loss value.
    """
    normal_true, multilabel_true = y_true[:, 0:1], y_true[:, 1:]
    normal_pred, multilabel_pred = y_pred[0], y_pred[1]

    # Loss for normal-abnormal classification
    normal_loss = tf.keras.losses.binary_crossentropy(normal_true, normal_pred)
    normal_loss = tf.reduce_mean(normal_loss)
    # Mask: Only compute multilabel loss for abnormal images
    mask = 1.0 - tf.cast(normal_true, tf.float32)  # 1 if abnormal, 0 if normal
    multilabel_loss = tf.keras.losses.binary_crossentropy(
        multilabel_true, multilabel_pred
    )
    masked_multilabel_loss = tf.reduce_mean(mask * multilabel_loss, axis=-1)
    masked_multilabel_loss = tf.reduce_mean(masked_multilabel_loss)
    # Combine the losses
    total_loss = normal_loss + rho * masked_multilabel_loss
    logger.info(
        "The normal_loss is %s; the masked_multilabel_loss is %s; the total_loss is %s ",
        str(normal_loss.numpy()),
        str(masked_multilabel_loss.numpy()),
        str(total_loss.numpy()),
    )
    return total_loss
