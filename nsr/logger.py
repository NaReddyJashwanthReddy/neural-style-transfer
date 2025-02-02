import os
from datetime import datetime
import logging

LOG_FILE=f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs")
if not os.path.exists(logs_path):
    os.mkdir(logs_path)

logs_file_path=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=logs_file_path,
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
