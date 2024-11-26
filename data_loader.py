import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from config import IMAGE_INFO, IMAGE_PATH
import os


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


def create_datasets(
    batch_size,
    image_size=(512, 512),
    test_set_ratio=0.2,
    random_seed=99,
):
    data_df = pd.read_excel(IMAGE_INFO, sheet_name="Sheet1")
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

    def create_tf_dataset(left_paths, right_paths, labels):
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
        dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    train_dataset = create_tf_dataset(train_left, train_right, train_labels)
    test_dataset = create_tf_dataset(test_left, test_right, test_labels)
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = create_datasets(batch_size=32)
    for train_batch, train_labels in train_dataset.take(1):
        print("Training batch shape:", train_batch)  # (batch_size, height, width, 6)
        print("Training labels shape:", train_labels)
