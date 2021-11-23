import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=None):
        self._best_loss = np.inf
        self._patience = patience
        self._num_since_best = 0
    
    def report_loss(self, loss):
        # return: stop?, is_best_so_far
        if self._patience is None:
            return False, False

        if loss <= self._best_loss:
            self._best_loss = loss
            self._num_since_best = 0
            return False, True
        
        self._num_since_best += 1
        return self._num_since_best >= self._patience, False

def create_input_dict(data_point, use_cuda):
    if isinstance(data_point, (tuple, list)):
        x, label = data_point
        data_point = {'x': x, 'label': label}

    if use_cuda:
        for data_val_name, data_val in data_point.items():
            if torch.is_tensor(data_val):
                data_point[data_val_name] = data_val.cuda()

    return data_point
