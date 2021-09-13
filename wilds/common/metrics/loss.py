import torch
from wilds.common.utils import avg_over_groups, maximum
from wilds.common.metrics.metric import ElementwiseMetric, Metric, MultiTaskMetric

class Loss(Metric):
    def __init__(self, loss_fn, name=None):
        self.loss_fn = loss_fn
        if name is None:
            name = 'loss'
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        """
        Helper for computing element-wise metric, implemented for each metric
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - element_wise_metrics (Tensor): tensor of size (batch_size, )
        """
        return self.loss_fn(y_pred, y_true)

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (float): Worst-case metric
        """
        return maximum(metrics)
    
class ElementwiseLoss(ElementwiseMetric):
    def __init__(self, loss_fn, name=None):
        self.loss_fn = loss_fn
        if name is None:
            name = 'loss'
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true, vocab_size):
        """
        Helper for computing element-wise metric, implemented for each metric
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - element_wise_metrics (Tensor): tensor of size (batch_size, )
        """
        #print(f"ypred T: {y_pred[0]}")
        #print(f"*************")
        #print(f"{y_true}")
        #print(f"ypred: {y_pred[0].shape}")
        #print(f"ytrue: {y_true.shape}")
        crazy_pred = y_pred[0].view(-1, vocab_size) #FIXME VOCAB SIZE
        crazy_true = y_true.view(-1)
        return self.loss_fn(crazy_pred, crazy_true)

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (float): Worst-case metric
        """
        return maximum(metrics)

class MultiTaskLoss(MultiTaskMetric):
    def __init__(self, loss_fn, name=None):
        self.loss_fn = loss_fn # should be elementwise
        if name is None:
            name = 'loss'
        super().__init__(name=name)

    def _compute_flattened(self, flattened_y_pred, flattened_y_true):
        if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            flattened_y_pred = flattened_y_pred.float()
            flattened_y_true = flattened_y_true.float()
        flattened_loss = self.loss_fn(flattened_y_pred, flattened_y_true)
        return flattened_loss

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (float): Worst-case metric
        """
        return maximum(metrics)

