import sys
from loguru import logger
from pathlib import Path
from datetime import datetime

# set up logging
logger.remove()
logger.add(sys.stdout, level='INFO')
current_time = datetime.now()
logger.add(Path(f"./logs/log_{current_time.strftime('%d-%m-%YT%H_%M_%S')}.log"))