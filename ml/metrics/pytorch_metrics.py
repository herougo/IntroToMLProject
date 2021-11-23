import torch

from ml.metrics.meter_metrics import AverageMeter
from ml.metrics.pytorch_func_metrics import cls_accuracy


class ClassAccuracyMetric:
    def __init__(self):
        self._meter = AverageMeter()
    
    def __call__(self,
                 batch_preds: torch.Tensor,
                 batch_targets: torch.Tensor):
        acc = cls_accuracy(batch_preds, batch_targets)[0]
        acc = acc.item()
        batch_size = batch_preds.size()[0]
        self._meter.update(acc, batch_size)
        
    def get_metric(self, reset=False):
        result = self._meter.val
        if reset:
            self.reset()
        return result
        
    def reset(self) -> None:
        self._meter.reset()