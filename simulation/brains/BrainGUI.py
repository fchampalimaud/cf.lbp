import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from brain_base import DataBrain, ChoiceParam


def _networks_base():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'networks'))


def _project_choices():
    base = _networks_base()
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))


def _network_choices():
    base = _networks_base()
    if not os.path.isdir(base):
        return []
    return sorted(f for f in os.listdir(base) if f.endswith('.json'))


class BrainGUI(DataBrain):
    """
    Generic data-driven brain. Loads a circuit from a JSON file in networks/.
    Design circuits graphically in the Network Visualizer and save them there.

    Set a project directory to organise networks into sub-folders of networks/.
    When a project is selected the file list shows only files in that sub-folder.
    """
    network_project = ChoiceParam(_project_choices, default='', desc='Project directory')
    network_file    = ChoiceParam(_network_choices, default='', desc='Network to run')

    # Class-level empty lists — overridden per-instance after network load
    sensors     = []
    layers      = []
    connections = []

    def setup(self):
        self.mL = self.mR = 0.0

    def loop(self, dt):
        self.step_network(dt)
        motor = getattr(self, 'motor', None)
        if motor is not None and motor.output is not None:
            self.mL = float(motor.output[0])
            self.mR = float(motor.output[1])
        return self.mL, self.mR
