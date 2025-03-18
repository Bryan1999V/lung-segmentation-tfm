"""Main script for the Pytorch module."""

import logging

import torch

from .basic_features import basic_features

DEFAULT_LOG_DATE_FORMAT = "%d-%m-%Y %H:%M:%S"
DEFAULT_LOG_FORMAT = "[%(asctime)s][%(module)s][%(levelname)s] > %(message)s"
DEFAULT_RANDOM_SEED_GENERATOR = 42

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# Set seed number
if torch.cuda.is_available():
    torch.cuda.manual_seed(DEFAULT_RANDOM_SEED_GENERATOR)


def main() -> None:
    """Run the main application of the Pytorch tutorial module."""
    logging.basicConfig(level=logging.DEBUG, format=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_LOG_DATE_FORMAT)
    basic_features()


if __name__ == "__main__":
    main()
