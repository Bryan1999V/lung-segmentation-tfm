"""Main script for the Pytorch module."""

import logging
import torch

from .basic_features import basic_features

DEFAULT_LOG_DATE_FORMAT = "%d-%m-%Y %H:%M:%S"
DEFAULT_LOG_FORMAT = "[%(asctime)s][%(levelname)s] > %(message)s"
DEFAULT_RANDOM_SEED_GENERATOR = 42

log = logging.getLogger(__name__)

# Set seed number
torch.manual_seed(DEFAULT_RANDOM_SEED_GENERATOR)


def main() -> None:
    """Main function of the Pytorch tutorial module."""
    logging.basicConfig(level=logging.DEBUG, format=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_LOG_DATE_FORMAT)
    basic_features()


if __name__ == "__main__":
    main()
