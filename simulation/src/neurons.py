"""
neurons.py — public aggregator.

Import from this module as before; the implementation lives in:
  neurons_base      — DynamicsBase, LayerBase, ACTIVATIONS, OUTPUT_MODES, _activate
  neurons_dynamics  — LeakyLayer, ProductLayer, AccumulatorLayer, AdaptiveLayer, MatsuokaLayer,
                      PulseLayer, SineLayer, RingAttractorLayer
  neurons_simple    — ConstantLayer, SumLayer, MotorLayer
  neurons_vision    — Conv2dLayer, Leaky2dLayer, Reichardt2dLayer
  neurons_learning  — LearningLayerBase, TDLayer, DeltaLayer, ThreeFactorLayer, SnapshotLayer
"""

from neurons_base import (
    ACTIVATIONS, OUTPUT_MODES, _activate,
    DynamicsBase, LayerBase,
)
from neurons_dynamics import (
    LeakyLayer, ProductLayer, AccumulatorLayer, AdaptiveLayer, MatsuokaLayer,
    PulseLayer, SineLayer, RingAttractorLayer,
)
from neurons_simple import (
    ConstantLayer, SumLayer, MotorLayer,
)
from neurons_vision import (
    Conv2dLayer, Leaky2dLayer, Reichardt2dLayer,
)
from neurons_learning import (
    LearningLayerBase, TDLayer, DeltaLayer, ThreeFactorLayer, SnapshotLayer,
)

# Registry populated automatically by LayerBase.__init_subclass__ as each class is defined.
LAYER_REGISTRY = LayerBase._registry
