# COPYRIGHT 2021. Fred Fung. Boston University.
import abc
from typing import Any

import torch

from utils.logging import get_logger


class SiameseTracker(torch.nn.Module):

    def __init__(self, cfg):
        super(SiameseTracker, self).__init__()
        self.cfg = cfg
        self._logger = get_logger(__name__)

    @abc.abstractmethod
    def forward(self, data):
        pass

    @abc.abstractmethod
    def get_trainable_weights(self, train_backbone):
        pass

    @abc.abstractmethod
    def compute_loss(self, data):
        pass

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
