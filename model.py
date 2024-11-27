import tensorflow as tf
from keras.applications import InceptionV3
from keras import layers, models, Model
import logging
from keras.applications import (
    InceptionV3,
    ResNet50,
    ResNet101,
    DenseNet121,
    VGG16,
    EfficientNetB0,
)  # Import other models as needed


logger = logging.getLogger(__name__)


class MyCustomModel(Model):
    def __init__(
        self, image_size=(512, 512), num_classes=8, base_model_name="MySimpleModel"
    ):
        super(MyCustomModel, self).__init__()
        self.base_model_name = base_model_name
        base_model_class = self._get_base_model_class(base_model_name)
        self.base_model = base_model_class(
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

    @staticmethod
    def _get_base_model_class(base_model_name):
        """
        Retrieves the class for the specified base model.

        Parameters:
            base_model_name (str): The name of the base model.

        Returns:
            class: The base model class from tf.keras.applications.

        Raises:
            ValueError: If the specified model name is not supported.
        """
        available_models = {
            "InceptionV3": InceptionV3,
            "ResNet50": ResNet50,
            "ResNet101": ResNet101,
            "DenseNet121": DenseNet121,
            "VGG16": VGG16,
            "EfficientNetB0": EfficientNetB0,
            "MySimpleModel": MySimpleModel,
        }

        if base_model_name not in available_models:
            raise ValueError(
                f"Unsupported base model name '{base_model_name}'. "
                f"Available options are: {list(available_models.keys())}."
            )
        return available_models[base_model_name]

    def call(self, inputs):
        left_x = self.base_model(inputs[:, :, :, :3])
        right_x = self.base_model(inputs[:, :, :, 3:])
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

    def __init__(self, num_classes=8, **kwargs):
        super(MySimpleModel, self).__init__()

        # Shared backbone layers
        self.conv1 = layers.Conv2D(32, (3, 3), activation="relu")
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(num_classes, (3, 3), activation="relu")
        self.pool2 = layers.MaxPooling2D((2, 2))

    def call(self, inputs):
        # Shared backbone computation
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        return x


if __name__ == "__main__":

    input_shape = (512, 512, 6)
    num_classes = 8
    model = MyCustomModel(
        image_size=(512, 512), num_classes=num_classes, base_model_name="InceptionV3"
    )
    tmp = tf.zeros((32, 512, 512, 6), dtype=tf.float32)
    zz = model(tmp)

    print(zz[0].shape, zz[1].shape)
