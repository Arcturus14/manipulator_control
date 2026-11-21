"""The same D0 projections on the explicit-replay execution backend."""
from controllers.dynamic_bound_mppi import DynamicBoundMixin
from controllers.captured_final_mppi import CapturedFinalTracker


class DynamicBoundCapturedTracker(DynamicBoundMixin, CapturedFinalTracker):
    pass
