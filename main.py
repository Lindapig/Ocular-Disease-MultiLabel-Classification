import tensorflow as tf
from logging_config import setup_logging
from model import MyCustomModel
from config import LOG_PATH, SAVED_MODEL_PATH
from data_loader import create_datasets
from training import train_model
import logging
import time
import argparse


start_time = time.time()
setup_logging(folder_path=LOG_PATH)
logger = logging.getLogger(__name__)


# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a two-stage model with specified configurations."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epoch", type=int, default=60, help="Number of training epochs."
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=256,
        help="Image size height.",
    )
    parser.add_argument(
        "--image_weight",
        type=int,
        default=256,
        help="Image size weight.",
    )
    parser.add_argument(
        "--test_set_ratio", type=float, default=0.2, help="Ratio of the test set."
    )
    parser.add_argument(
        "--random_seed", type=int, default=99, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate for optimizer.",
    )
    parser.add_argument(
        "--rho", type=float, default=3.0, help="Rho value for training configuration."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MySimpleModel",
        help="Only support MySimpleModel, InceptionV3,ResNet50,ResNet101,DenseNet121,VGG16,EfficientNetB0",
    )
    return parser.parse_args()


# Main function
def main():
    start_time = time.time()

    # Parse command-line arguments
    args = parse_args()
    image_size = (args.image_height, args.image_weight)
    # Log configuration details
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of epochs: {args.num_epoch}")
    logger.info(f"Image size: {image_size}")
    logger.info(f"Test set ratio: {args.test_set_ratio}")
    logger.info(f"Random seed: {args.random_seed}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Rho: {args.rho}")
    logger.info(f"Model {args.model}")

    # Create datasets
    train_dataset, test_dataset = create_datasets(
        image_size=image_size,
        batch_size=args.batch_size,
        test_set_ratio=args.test_set_ratio,
        random_seed=args.random_seed,
    )

    # Initialize and train the model
    model = MyCustomModel(
        image_size=image_size,
        num_classes=8,
        base_model_name=args.model,  # "MySimpleModel",
    )
    model = train_model(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        learning_rate=args.learning_rate,
        epochs=args.num_epoch,
        rho=args.rho,
    )

    # Log running time
    logger.info(
        f"Training completed. Total running time: {time.time() - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()
