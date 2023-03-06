import math
import random
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from torch.optim import Adam, SGD, lr_scheduler
import numpy as np
import torch.nn as nn
from utils.losses import hybrid_loss, jaccard_loss, dice_loss
import datetime
import logging
import os
import torch
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image


def find_gpu():
    nvmlInit()
    mem = []
    nvidia_count = nvmlDeviceGetCount()
    for i in range(nvidia_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        mem.append(memory_info.free)

    index = np.where(np.array(mem) > 5000000000)[0]
    gpu_index = index[-1]
    return str(gpu_index)


def save_images(images, path, names):
    images = torch.chunk(images, images.shape[0])
    for index, image in enumerate(images):
        save_image(image, path + "stylized_" + names[index])


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_criterion(loss_function):
    """get the user selected loss function
    Parameters
    ----------
    opt : dict
        Dictionary of options/flags
    Returns
    -------
    method
        loss function
    """
    if loss_function == 'hybrid':
        criterion = hybrid_loss
    if loss_function == 'bce':
        criterion = nn.CrossEntropyLoss()
    if loss_function == 'dice':
        criterion = dice_loss
    if loss_function == 'jaccard':
        criterion = jaccard_loss

    return criterion


def get_logger(logdir):
    logger = logging.getLogger(__name__)
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # console.setFormatter(formatter)
    # logger.addHandler(console)
    return logger


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, score):

        # score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(score, model)
            self.counter = 0
