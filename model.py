import tensorflow as tf
from keras.applications import InceptionV3
from keras import layers, models, Model
import logging

logger = logging.getLogger(__name__)


class MyInceptionV3(Model):
    def __init__(self, image_size=(512, 512), num_classes=8):
        super(MyInceptionV3, self).__init__()

        self.inception_v3 = InceptionV3(
            weights="imagenet",
            include_top=False,
            input_shape=image_size + (3,),
            classes=1000,
            classifier_activation=None,
        )
        self.flatten = layers.Flatten()
        self.dense_shared = layers.Dense(128, activation="relu")
        # Output layers for each stage
        self.normal_abnormal_output = layers.Dense(
            1, activation="sigmoid", name="normal_abnormal"
        )
        self.multilabel_output = layers.Dense(
            num_classes - 1, activation="sigmoid", name="multilabel"
        )

    def call(self, inputs):
        left_x = self.inception_v3(inputs[:, :, :, :3])
        right_x = self.inception_v3(inputs[:, :, :, 3:])
        # Shared backbone computation
        left_x = self.flatten(left_x)
        right_x = self.flatten(right_x)
        x = tf.concat([left_x, right_x], axis=-1)
        features = self.dense_shared(x)
        # Stage 1: Normal-Abnormal Classification
        normal_abnormal = self.normal_abnormal_output(features)

        # Stage 2: Multilabel Classification
        multilabel = self.multilabel_output(features)

        return normal_abnormal, multilabel


class MySimpleModel(Model):
    """

    Simple Two-stage model:
    1. Stage 1: Classify as normal or abnormal.
    2. Stage 2: Multilabel classification for abnormal images.

    Args:
    - input_shape: Shape of the input image (H, W, C).
    - num_classes: Number of disease classes (excluding 'normal').
    """

    def __init__(self, num_classes=8):
        super(MySimpleModel, self).__init__()

        # Shared backbone layers
        self.conv1 = layers.Conv2D(32, (3, 3), activation="relu")
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation="relu")
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense_shared = layers.Dense(128, activation="relu")

        # Output layers for each stage
        self.normal_abnormal_output = layers.Dense(
            1, activation="sigmoid", name="normal_abnormal"
        )
        self.multilabel_output = layers.Dense(
            num_classes - 1, activation="sigmoid", name="multilabel"
        )

    def call(self, inputs):
        # Shared backbone computation
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        features = self.dense_shared(x)

        # Stage 1: Normal-Abnormal Classification
        normal_abnormal = self.normal_abnormal_output(features)

        # Stage 2: Multilabel Classification
        multilabel = self.multilabel_output(features)

        return normal_abnormal, multilabel


if __name__ == "__main__":

    input_shape = (512, 512, 6)
    num_classes = 8
    model = MyInceptionV3(image_size=(512, 512), num_classes=num_classes)
    tmp = tf.zeros((32, 512, 512, 6), dtype=tf.float32)
    zz = model(tmp)
    print(zz[0].shape, zz[1].shape)
