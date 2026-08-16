"""Common interface for MPPI cost components."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseCost(ABC):
    @abstractmethod
    def running(self, state: torch.Tensor, action: torch.Tensor, step: int) -> torch.Tensor:
        """Compute the batched stage cost."""

    @abstractmethod
    def terminal(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute the batched terminal cost."""
