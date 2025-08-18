import logging
from datetime import datetime
from pathlib import Path

now = datetime.now().strftime("%d%m%Y-%H:%M:%S")
log_dir = Path('logs')
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f'{now}.log'

logging.basicConfig(
        filename=log_path,
        encoding='utf-8',
        format='%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s',
        datefmt='%d%m%Y %H:%M:%S',
        level=20

        )
logger = logging.getLogger(__name__)

def test_logger():
    logger.debug("Debug logs shouldn't appear")
    logger.info('Test Info Log')
    logger.warning('Test Warning Log')
    logger.error('Test Error Log')
    pass

if __name__ == "__main__":
    test_logger()
