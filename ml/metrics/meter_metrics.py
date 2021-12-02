import numpy as np

# copied: https://github.com/moemen95/Pytorch-Project-Template/blob/master/utils/metrics.py

class AverageMeter:
    """
    Class to be an average meter for any average metric like loss, accuracy, etc..
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_metric(self, reset=False):
        result = self.val
        if reset:
            self.reset()
        return result

    @property
    def val(self):
        return self.avg


class AverageMeterList:
    """
    Class to be an average meter for any average metric List structure like mean_iou_per_class
    """

    def __init__(self, num_cls):
        self.cls = num_cls
        self.reset()

    def reset(self):
        self.value = [0] * self.cls
        self.avg = [0] * self.cls
        self.sum = [0] * self.cls
        self.count = [0] * self.cls

    def update(self, val, n=1):
        for i in range(self.cls):
            self.value[i] = val[i]
            self.sum[i] += val[i] * n
            self.count[i] += n
            self.avg[i] = self.sum[i] / self.count[i]

    @property
    def val(self):
        return self.avg