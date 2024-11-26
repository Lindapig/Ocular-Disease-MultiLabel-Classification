import tensorflow as tf
from loss import custom_loss
from metrics import custom_evaluation
import logging
import os

logger = logging.getLogger(__name__)


def train_two_stage_model(
    model,
    train_dataset,
    test_dataset,
    learning_rate=0.001,
    epochs=10,
    save_model_path=None,
):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
    )
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        logger.info("Epoch is %s", str((epoch + 1) / epochs))
        train_loss = 0.0
        batch_count = 0

        # Training loop
        for images, labels in train_dataset:
            with tf.GradientTape() as tape:
                normal_pred, multilabel_pred = model(images, training=True)
                loss = custom_loss(labels, [normal_pred, multilabel_pred])
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss += loss.numpy()
            batch_count += 1

        train_loss /= batch_count
        print(f"Training Loss: {train_loss:.4f}")
        logger.info("training loss is %s", str(train_loss))
        # Evaluation loop
        test_loss = 0.0
        test_normal_accuracy = 0.0
        test_multilabel_accuracy = 0.0
        test_overall_accuracy = 0.0
        test_overall_label_wise_accuracy = 0.0
        batch_count = 0
        for images, labels in test_dataset:
            normal_pred, multilabel_pred = model(images, training=False)
            loss = custom_loss(labels, [normal_pred, multilabel_pred])
            test_loss += loss.numpy()
            (
                normal_accuracy,
                multilabel_accuracy,
                overall_accuracy,
                overall_label_wise_accuracy,
            ) = custom_evaluation(labels, [normal_pred, multilabel_pred])
            test_normal_accuracy += normal_accuracy
            test_multilabel_accuracy += multilabel_accuracy
            test_overall_accuracy += overall_accuracy
            test_overall_label_wise_accuracy += overall_label_wise_accuracy
            batch_count += 1

        test_loss /= batch_count
        test_normal_accuracy /= batch_count
        test_multilabel_accuracy /= batch_count
        test_overall_accuracy /= batch_count
        test_overall_label_wise_accuracy /= batch_count

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
        if save_model_path:
            model.save(os.path.join(save_model_path, "my_model"))
    return model
