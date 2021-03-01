# -*- coding: utf-8 -*-
import logging
from logging import handlers
from datetime import datetime
from src.config import log_path
import os

now = datetime.now().strftime('%m%d%H%M')
now = now[:4] + "_" + now[4:]

def create_logger():
    log_path_ = os.path.join(log_path, now + '.log')
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    logger = logging.getLogger(log_path_)
    fmt = '%(asctime)s - %(levelname)s: %(message)s'
    datefmt = "%m-%d %H:%M:%S"
    format_str = logging.Formatter(fmt, datefmt=datefmt)
    logger.setLevel(level_relations.get('info'))
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = handlers.TimedRotatingFileHandler(
        filename=log_path_, when='D', backupCount=3,
        encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)

    return logger


logger = create_logger()
