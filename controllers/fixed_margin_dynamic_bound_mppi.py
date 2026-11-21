"""The same D0 projection with invariant-margin execution optimization."""
from controllers.dynamic_bound_mppi import DynamicBoundMixin
from controllers.fixed_margin_captured_mppi import FixedMarginCapturedTracker


class DynamicBoundFixedMarginCapturedTracker(DynamicBoundMixin, FixedMarginCapturedTracker):
    pass
