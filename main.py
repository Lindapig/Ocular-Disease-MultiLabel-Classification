import tensorflow as tf
from logging_config import setup_logging
from model import MySimpleModel, MyInceptionV3
from config import LOG_PATH, SAVED_MODEL_PATH
from data_loader import create_datasets
from training import train_two_stage_model
import logging
import time

start_time = time.time()
setup_logging(folder_path=LOG_PATH)
logger = logging.getLogger(__name__)


batch_size = 16
num_epoch = 10
image_size = (512, 512)
test_set_ratio = 0.2
random_seed = 99
learning_rate = 0.0001
save_model_path = SAVED_MODEL_PATH


logger.info(f"Batch size is {batch_size}")
logger.info(f"Number of epochs is {num_epoch}")
logger.info(f"Image size is {image_size}")
logger.info(f"Test set ratio is {test_set_ratio}")
logger.info(f"Random seed is {random_seed}")
logger.info(f"Model will be saved to {save_model_path}")
logger.info(f"Learning rate {learning_rate}")
train_dataset, test_dataset = create_datasets(
    batch_size=batch_size,
    test_set_ratio=test_set_ratio,
    random_seed=random_seed,
)

model = MyInceptionV3(
    image_size=image_size, num_classes=8
)  # the num_classes includes normal
# model = MySimpleModel(num_classes=8)
model = train_two_stage_model(
    model,
    train_dataset,
    test_dataset,
    learning_rate=learning_rate,
    epochs=num_epoch,
    save_model_path=save_model_path,
)
logger.info(f"running time is {time.time()-start_time}")
