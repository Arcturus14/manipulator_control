"""Common interface for MPPI cost components."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseCost(ABC):
    # pytorch_mppi calls the stage cost *after* the dynamics: at stage t it
    # receives (x(t+1), u(t)). Time-indexed state references therefore use
    # t+1, while input terms keep t. Zero reproduces results recorded before
    # this alignment.
    reference_offset: int = 1

    def state_index(self, step: int) -> int:
        """Horizon time index of the state passed to stage ``step``."""
        return step + self.reference_offset

    @abstractmethod
    def running(self, state: torch.Tensor, action: torch.Tensor, step: int) -> torch.Tensor:
        """Compute the batched stage cost."""

    @abstractmethod
    def terminal(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute the batched terminal cost."""
