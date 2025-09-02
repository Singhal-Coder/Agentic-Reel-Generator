import logging
import coloredlogs
from logging.handlers import RotatingFileHandler

from ..config.settings import settings

def setup_logging(log_level=settings.LOG_LEVEL.upper()):
    """
    Sets up a robust, visually enhanced logger with rotation and colored output.
    """
    logger = logging.getLogger("reel_generator_ai_module")
    
    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(log_level)

    # Console handler with colors
    log_format_console = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    coloredlogs.install(
        level=log_level,
        logger=logger,
        fmt=log_format_console,
        level_styles={
            'debug': {'color': 'green'},
            'info': {'color': 'cyan'},
            'warning': {'color': 'yellow'},
            'error': {'color': 'red', 'bold': True},
            'critical': {'color': 'red', 'bold': True, 'background': 'white'}
        }
    )

    # File handler with rotation
    log_dir = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'ai_module.log'

    file_handler = RotatingFileHandler(
        log_path, 
        maxBytes=5*1024*1024,  # 5 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG) # Log all levels to file
    
    log_format_file = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    file_handler.setFormatter(log_format_file)
    logger.addHandler(file_handler)

    return logger

logger = setup_logging()