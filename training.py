import tensorflow as tf
from loss import custom_loss
from metrics import custom_accuracy, f_1_score, auc_roc, plot_confusion_matrix
from config import SAVED_MODEL_PATH
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


def train_model(
    model,
    train_dataset,
    test_dataset,
    learning_rate=0.001,
    epochs=10,
    rho=3,
):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
    )
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        logger.info("Epoch is %s", str((epoch + 1)) + "/" + str(epochs))
        train_loss = 0.0
        batch_count = 0

        # Training loop
        for images, labels in train_dataset:
            with tf.GradientTape() as tape:
                normal_pred, multilabel_pred = model(images, training=True)
                loss = custom_loss(labels, [normal_pred, multilabel_pred], rho=rho)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss += loss.numpy()
            batch_count += 1

        train_loss /= batch_count
        print(f"Training Loss: {train_loss:.4f}")
        logger.info("training loss is %s", str(train_loss))
        # Evaluation loop
        test_loss = 0.0
        batch_count = 0

        y_true_list = []
        y_pred_probs_list = []
        for images, labels in test_dataset:
            normal_pred, multilabel_pred = model(images, training=False)
            loss = custom_loss(labels, [normal_pred, multilabel_pred], rho=rho)
            test_loss += loss.numpy()
            batch_count += 1
            y_true_list.append(labels.numpy())
            y_pred_probs_tmp = tf.concat([normal_pred, multilabel_pred], axis=-1)
            y_pred_probs_list.append(y_pred_probs_tmp.numpy())

        test_loss /= batch_count

        y_true = np.concatenate(y_true_list, axis=0)
        y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)

        (
            test_normal_accuracy,
            test_multilabel_accuracy,
            test_overall_accuracy,
            test_overall_label_wise_accuracy,
        ) = custom_accuracy(y_true, y_pred_probs, threshold=0.5)
        if epoch % 10 == 0 and epoch != 0:
            plot_confusion_matrix(
                y_true,
                y_pred_probs,
                threshold=0.5,
                epoch=epoch,
                model_name=model.base_model_name,
            )
        f_1 = f_1_score(y_true, y_pred_probs, threshold=0.5)
        ac = auc_roc(y_true, y_pred_probs)
        print(
            f"Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_overall_accuracy * 100:.2f}%"
        )
        logger.info("The test_loss is %s", str(test_loss))
        logger.info("test_normal_accuracy is %s", str(test_normal_accuracy))
        logger.info("test_multilabel_accuracy is %s", str(test_multilabel_accuracy))
        logger.info("test_overall_accuracy is %s", str(test_overall_accuracy))
        logger.info(
            "test_overall_label_wise_accuracy is %s",
            str(test_overall_label_wise_accuracy),
        )
        logger.info(
            "F 1 score is %s",
            str(f_1),
        )
        logger.info(
            "AUC ROC is %s",
            str(ac),
        )
        if SAVED_MODEL_PATH:
            if epoch % 5 == 0 and epoch != 0:
                model.save(
                    os.path.join(
                        SAVED_MODEL_PATH,
                        f"{model.base_model_name}_{epoch}",
                    )
                )
    return model
