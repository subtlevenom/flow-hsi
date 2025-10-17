import sys
import logging


def logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger
