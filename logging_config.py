import logging
from datetime import datetime
import pytz
import os

# LOGGING_PATH = f"/opt/omniai/work/instance1/jupyter/mlfinlib/mlfinlib/playground/log/log_{timestamp}"


def setup_logging(
    folder_path="/media/wang/data/proj/ODIR/log/",
    prefix="log",
):
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"{prefix}_{timestamp}.log"
    filename = os.path.join(folder_path, filename)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
        handler.close()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d",
        filename=filename,
    )
