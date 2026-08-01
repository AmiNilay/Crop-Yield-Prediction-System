import logging
import os
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
#  Project-relative log directory (not os.getcwd())
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
_LOGS_DIR: Path = _PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FILE: str = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
_LOG_FILE_PATH: Path = _LOGS_DIR / _LOG_FILE

logging.basicConfig(
    filename=str(_LOG_FILE_PATH),
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Also log to console for development
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.WARNING)
_console_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
)
logging.getLogger().addHandler(_console_handler)

if __name__ == "__main__":
    logging.info("Logging started -> %s", _LOG_FILE_PATH)