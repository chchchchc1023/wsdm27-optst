import random
import numpy as np
import torch
import logging
import os
from datetime import datetime


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_logger(log_dir="./exp", exp_name=None):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_dir = os.path.join(log_dir, exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    log_path = os.path.join(exp_dir, f"training_{exp_name}.log")

    logger = logging.basicConfig(
        filename=log_path,
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filemode='w'
    )

    return exp_dir
