"""Unchanged dynamic-bound comparator on the fully captured execution backend."""
from controllers.dynamic_bound_mppi import DynamicBoundMixin
from controllers.fully_captured_final_mppi import FullyCapturedFinalTracker


class DynamicBoundFullyCapturedTracker(DynamicBoundMixin, FullyCapturedFinalTracker):
    pass
