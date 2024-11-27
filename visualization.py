import matplotlib.pyplot as plt
import math
import tensorflow as tf
import numpy as np


def visualize_batch(dataset, batch_size=32, image_size=(512, 512)):
    """
    Visualize a batch of images from a TensorFlow dataset in a single plot.

    Parameters:
        dataset: tf.data.Dataset - The dataset to visualize.
        batch_size: int - The number of images in the batch.
        image_size: tuple - The size of the images for display.
    """
    for batch in dataset.take(1):  # Take a single batch
        images, labels = batch
        images = images.numpy()  # Convert images to numpy array
        labels = labels.numpy()  # Convert labels to numpy array

        # Calculate grid size
        grid_cols = 8  # Number of images per row
        grid_rows = math.ceil(batch_size / grid_cols)

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(16, 2 * grid_rows))

        for i in range(batch_size):
            row = i // grid_cols
            col = i % grid_cols

            # Split concatenated image into left and right parts
            left_image = images[i, :, :, :3]  # Assuming RGB channels for the left image
            right_image = images[
                i, :, :, 3:
            ]  # Assuming RGB channels for the right image

            # Combine left and right image for visualization
            combined_image = tf.concat([left_image, right_image], axis=1).numpy()
            combined_image = (combined_image * 255).astype(np.uint8)
            ax = axes[row, col]
            ax.imshow(combined_image)
            ax.set_title(f"Label: {labels[i]}")
            ax.axis("off")

        # Hide unused subplots
        for j in range(batch_size, grid_cols * grid_rows):
            fig.delaxes(axes.flat[j])

        plt.tight_layout()
        plt.show()
        break  # Only process one batch
