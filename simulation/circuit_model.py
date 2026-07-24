"""
circuit_model.py — lightweight container for the active circuit state.

Owned by SimulatorApp; shared (read/write) with NetworkVisualizerWindow.
Decouples the visualizer from the app's internal attribute namespace.
"""
from dataclasses import dataclass


@dataclass
class Connection:
    """A weighted connection between two named nodes in the circuit.

    src      : name of the source layer or sensor
    tgt      : name of the target layer
    W        : weight matrix (numpy array) or conv kernel (4-D array)
    learning : learning rule, or None for fixed weights.
               Supported: 'dopamine_hebbian'
    lr       : learning rate used by the active learning rule
    """
    src: str
    tgt: str
    W: object
    learning: str = None
    lr: float = 0.01
    init_W: object = None   # snapshot of W at the time the user set it via the dialog


class CircuitModel:
    """
    Holds the lists that define the active circuit at runtime:

    sensors     : list of BaseSensor instances (from brain class + config)
    layers      : list of neuron layer instances (LeakyLayer, etc.)
    connections : list of Connection objects
    bodies      : list of RigidBody — root body always at index 0
    joints      : list of Joint connecting bodies
    """
    __slots__ = ('sensors', 'layers', 'connections', 'bodies', 'joints')

    def __init__(self, sensors=None, layers=None, connections=None,
                 bodies=None, joints=None):
        self.sensors     = sensors     if sensors     is not None else []
        self.layers      = layers      if layers      is not None else []
        self.connections = connections if connections is not None else []
        self.bodies      = bodies      if bodies      is not None else []
        self.joints      = joints      if joints      is not None else []

    def clear(self):
        self.sensors.clear()
        self.layers.clear()
        self.connections.clear()
        self.bodies.clear()
        self.joints.clear()
