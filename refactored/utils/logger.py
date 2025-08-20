import logging
import os

def setup_logging(log_file: str = "/app/logs/streamlit_log.log"):
    """Setup application logging."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
