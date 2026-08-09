import numpy as np


class Param:
    def __init__(self, default, min_val, max_val, step=0.01, desc=""):
        self.default, self.min, self.max, self.step, self.desc = default, min_val, max_val, step, desc


class ChoiceParam:
    """String-valued param rendered as a QComboBox. choices may be a list or callable."""
    def __init__(self, choices, default='', desc=''):
        self.choices = choices
        self.default = default
        self.desc    = desc

    def get_choices(self):
        return self.choices() if callable(self.choices) else list(self.choices)


class BaseConfig:
    def __init__(self):
        for key, value in self.__class__.__dict__.items():
            if isinstance(value, Param) or hasattr(value, 'get_choices'):
                setattr(self, key, value.default)

    def get_param_metadata(self):
        return {k: v for k, v in self.__class__.__dict__.items()
                if isinstance(v, Param) or hasattr(v, 'get_choices')}


class BaseBrain(BaseConfig):
    def step_network(self, dt):
        from network_runner import step_network as _run
        _run(self, dt)

    def plots(self):
        return []


class DataBrain(BaseBrain):
    """Marker base class for data-driven brains that load circuits from JSON.

    Use isinstance(brain, DataBrain) instead of getattr(brain, '_is_data_brain', False).
    Subclasses (e.g. BrainGUI) provide a network_file ChoiceParam; the simulator
    loads the named JSON into the circuit when the brain is activated.
    """
