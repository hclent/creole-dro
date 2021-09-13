import time
from tqdm import tqdm
from math import floor

import torch
import sklearn

from wilds.common.utils import avg_over_groups, minimum, maximum
from wilds.common.metrics.metric import Metric

from utils import format_time


def perplexity(dataset_obj, test_loader, model, device):
    # TODO: figure out best numbers to use here. #Tutorial has max_length = 1024 and stride at 512
    max_length = dataset_obj.max_len
    stride = floor(max_length / 2)
    print(f"stride: {stride}")

    t0 = time.time()
    lls = []
    for step, batch in enumerate(test_loader):
        if step % 500 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_loader), elapsed))
        b_input_ids = batch["input_ids"]
        b_labels = batch["labels"]
        model.zero_grad()
        for i in tqdm(range(0, b_input_ids.size(1), stride)):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, b_input_ids.size(1))
            trg_len = end_loc - i
            input_ids = b_input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = model(input_ids, target_ids)
                log_likelihood = outputs[0] * trg_len
            lls.append(log_likelihood)

    perplexity = torch.exp(torch.stack(lls).sum() / end_loc)
    return perplexity


class F1(Metric):
    def __init__(self, prediction_fn, name=None, average='macro'):
        """

        :param prediction_fn: logits to prediction
        :param name:
        :param average:
        """
        self.prediction_fn = prediction_fn
        if name is None:
            name = f'F1'
        self.average = average
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        device = y_pred.device
        # score = sklearn.metrics.f1_score(y_true, y_pred, average=self.average, labels=torch.unique(y_true.cpu()))
        score = sklearn.metrics.f1_score(y_true.cpu(), y_pred.cpu(), average=self.average)
        return torch.tensor(score).to(device)

    def worst(self, metrics):
        return minimum(metrics)
