"""Model registry auto-discovery."""

from .lstm.model import LSTMModel  # noqa: F401
from .pcrnn.model import PCRNNModel  # noqa: F401
from .node.model import NODEModel  # noqa: F401
from .anode.model import ANODEModel  # noqa: F401
from .ude_charm.model import CHARMUDE  # noqa: F401
from .acla.model import ACLAModel  # noqa: F401

__all__ = [
    "LSTMModel",
    "PCRNNModel",
    "NODEModel",
    "ANODEModel",
    "CHARMUDE",
    "ACLAModel",
]
