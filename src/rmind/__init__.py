import operator
from importlib.metadata import version

from omegaconf import OmegaConf

__version__ = version(__package__ or __name__)

# Arithmetic for config interpolation (e.g. sequence_length = history + horizon),
# registered before Hydra resolves any config. __init__ runs once per process, so
# a single registration suffices (and a name clash should surface, not be masked).
OmegaConf.register_new_resolver("add", operator.add)  # noqa: RUF067

__all__ = ["__version__"]
