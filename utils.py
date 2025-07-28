import datetime
import json
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import sys
import logging


# convert data from cpu to gpu, accelerate the running speed
def convert_to_gpu(data, device=None):
    if device != -1 and torch.cuda.is_available():
        data = data.to(device)
    return data


def convert_all_data_to_gpu(*data, device=None):
    res = []
    for item in data:
        item = convert_to_gpu(item, device=device)
        res.append(item)
    return tuple(res)


def convert_train_truth_to_gpu(train_data, truth_data):
    train_data = [[convert_to_gpu(basket) for basket in baskets] for baskets in train_data]
    truth_data = convert_to_gpu(truth_data)
    return train_data, truth_data


# load parameters of model
def load_model(model_object, model_file_path, map_location=None):
    if map_location is None:
        model_object.load_state_dict(torch.load(model_file_path))
    else:
        model_object.load_state_dict(torch.load(model_file_path, map_location=map_location))

    return model_object


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur


def set_logger(log_path, log_name='rec', mode='a'):
    """set up log file
    mode : 'a'/'w' mean append/overwrite,
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path, mode=mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, logger, patience=10, verbose=False, delta=0.01):
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_metric = None
        self.early_stop = False
        self.delta = delta
        self.logger = logger

    def __call__(self, val_metric_anchor, model):
        if self.best_val_metric is None:
            self.best_val_metric = val_metric_anchor
            self.save_checkpoint(val_metric_anchor, model)
        elif val_metric_anchor < self.best_val_metric:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val_metric = val_metric_anchor
            self.save_checkpoint(val_metric_anchor, model)
            self.counter = 0

    def save_checkpoint(self, val_metric_anchor, model):
        if self.verbose:
            self.logger.info(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)

