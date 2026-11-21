"""GUI baseline with the confirmed N0 control semantics, also usable with DOB OFF."""
from kroc.position_input_headroom_study import PositionInputTracker
from controllers.mppi_dob_controller import UR5Tracker

class FinalN0Tracker(PositionInputTracker):
    def __init__(self,*args,**kwargs):
        kwargs.update(study_case='N0',headroom_weight=0.,unclamped_diagnostic=False)
        super().__init__(*args,**kwargs)

    def run(self,*args,**kwargs):
        if kwargs.get('use_dob',False):
            return super().run(*args,**kwargs)
        # Same N0 costs/sampler and actuator limits; bypass only the study's DOB-required guard.
        self.request_sample_fraction.clear();self.headroom_sample_cost.clear()
        self.sensor_model.reset();self.measured_state_history.clear()
        self.margin_history.clear();self.margins.reset()
        return UR5Tracker.run(self,*args,**kwargs)
