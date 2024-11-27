import tensorflow as tf
import pandas as pd
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from config import IMAGE_INFO, IMAGE_PATH
import os
from utils import read_file


# Function to load and preprocess a single image
def load_and_preprocess_image(filepath, image_size=(512, 512)):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image


def load_and_concatenate_images(left_path, right_path, image_size=(512, 512)):
    left_image = load_and_preprocess_image(left_path, image_size=image_size)
    right_image = load_and_preprocess_image(right_path, image_size=image_size)
    concatenated_image = tf.concat(
        [left_image, right_image], axis=-1
    )  # Concatenate along channel
    return concatenated_image


def augment_images(left_image, right_image, image_size):
    """
    Apply the same augmentation to both left and right images.
    """
    # Random rotation
    if tf.random.uniform([]) > 0.4:
        angle = tf.random.uniform([], -0.2, 0.2)  # Rotation angle in radians
        left_image = tfa.image.rotate(left_image, angle)
        right_image = tfa.image.rotate(right_image, angle)

    # Random horizontal flip
    if tf.random.uniform([]) > 0.2:
        left_image = tf.image.flip_left_right(left_image)
        right_image = tf.image.flip_left_right(right_image)

    # Random brightness
    if tf.random.uniform([]) > 0.5:
        delta = tf.random.uniform([], 0, 0.3)
        left_image = tf.image.adjust_brightness(left_image, delta)
        right_image = tf.image.adjust_brightness(right_image, delta)

    # Random contrast
    if tf.random.uniform([]) > 0.5:
        contrast_factor = tf.random.uniform([], 1, 1.6)
        left_image = tf.image.adjust_contrast(left_image, contrast_factor)
        right_image = tf.image.adjust_contrast(right_image, contrast_factor)

    # # Random hue_delta adjustment
    # if tf.random.uniform([]) > 0.6:
    #     hue_delta = tf.random.uniform([], -0.4, 0.4)
    #     left_image = tf.image.adjust_hue(left_image, hue_delta)
    #     right_image = tf.image.adjust_hue(right_image, hue_delta)

    # # Random saturation adjustment
    # if tf.random.uniform([]) > 0.6:
    #     saturation_factor = tf.random.uniform([], 0.2, 2)
    #     left_image = tf.image.adjust_saturation(left_image, saturation_factor)
    #     right_image = tf.image.adjust_saturation(right_image, saturation_factor)

    # Clip values to [0, 1]
    left_image = tf.clip_by_value(left_image, 0.0, 1.0)
    right_image = tf.clip_by_value(right_image, 0.0, 1.0)
    left_image = tf.image.resize(left_image, image_size)
    right_image = tf.image.resize(right_image, image_size)
    concatenated_image = tf.concat([left_image, right_image], axis=-1)  #
    return concatenated_image


def create_datasets(
    batch_size,
    image_size=(512, 512),
    test_set_ratio=0.2,
    random_seed=99,
    is_augment=True,
):
    data_df = read_file(IMAGE_INFO)

    left_paths = (
        data_df["Left-Fundus"].apply(lambda x: os.path.join(IMAGE_PATH, x)).tolist()
    )
    right_paths = (
        data_df["Right-Fundus"].apply(lambda x: os.path.join(IMAGE_PATH, x)).tolist()
    )
    labels = data_df[["N", "D", "G", "C", "A", "H", "M", "O"]].values
    train_left, test_left, train_right, test_right, train_labels, test_labels = (
        train_test_split(
            left_paths,
            right_paths,
            labels,
            test_size=test_set_ratio,
            random_state=random_seed,
        )
    )

    def create_tf_dataset(left_paths, right_paths, labels, is_training=True):
        dataset = tf.data.Dataset.from_tensor_slices((left_paths, right_paths, labels))
        dataset = dataset.map(
            lambda left, right, label: (
                load_and_concatenate_images(
                    left, right, image_size=image_size
                ),  # Process images
                tf.convert_to_tensor(label, dtype=tf.int64),  # Convert labels to tensor
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if is_training and is_augment:
            dataset = dataset.map(
                lambda image, label: (
                    augment_images(
                        image[:, :, :3], image[:, :, 3:], image_size
                    ),  # Split left and right
                    label,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    train_dataset = create_tf_dataset(
        train_left, train_right, train_labels, is_training=True
    )
    test_dataset = create_tf_dataset(
        test_left, test_right, test_labels, is_training=False
    )
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = create_datasets(batch_size=32)
    for train_batch, train_labels in train_dataset.take(1):
        print("Training batch shape:", train_batch[0])  # (batch_size, height, width, 6)
        # print("Training labels shape:", train_labels)
