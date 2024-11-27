import logging

logger = logging.getLogger(__name__)

# IMAGE_PATH = "/media/wang/data/proj/ODIR/data/ODIR-5K/Training Images/"
# IMAGE_INFO = "/media/wang/data/proj/ODIR/data/ODIR-5K/data.xlsx"
IMAGE_PATH = "/media/wang/data/proj/ODIR/data/preprocessed_images/"
IMAGE_INFO = "/media/wang/data/proj/ODIR/data/filtered_full_df.csv"
SAVED_MODEL_PATH = "/media/wang/data/proj/ODIR/saved_model/"
LOG_PATH = "/media/wang/data/proj/ODIR/log/"
PLOT_PATH = "/media/wang/data/proj/ODIR/plot/"

logger.info(f"IMAGE_INFO: {IMAGE_INFO}")
logger.info(f"IMAGE_PATH  {IMAGE_PATH}")
logger.info(f"SAVED_MODEL_PATH: {SAVED_MODEL_PATH}")
logger.info(f"PLOT_PATH: {PLOT_PATH}")
