import logging

COLOR_RESET = "\033[0m"
COLOR_BLUE = "\033[94m"  # Blue
COLOR_GREEN = "\033[92m"  # Green
COLOR_YELLOW = "\033[0;33m" # Yellow


class ColorFormatter(logging.Formatter):
    level_colors = {
        logging.DEBUG: COLOR_BLUE,
        logging.INFO: COLOR_GREEN,
        logging.WARNING: COLOR_YELLOW
    }

    def format(self, record):
        level_color = ColorFormatter.level_colors.get(record.levelno, COLOR_RESET)
        record.msg = f"{level_color}{record.msg}{COLOR_RESET}"
        return super().format(record)


handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))

# Configure logger
logger = logging.getLogger(__name__)
# TODO: make a production mode with logging.INFO mode
# logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False  # Avoid duplicate logs

def set_debug_mode(debug_mode: bool):
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

def print_log(msg: str):
    logger.info(msg)

def print_warning(msg: str):
    logger.warning(msg)

def forward_warning(error: Exception):
    logger.warning("Exception during runtime:", exc_info=error)

def print_debug_log(msg: str):
    logger.debug(msg)
