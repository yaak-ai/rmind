from importlib.metadata import version

from omegaconf import OmegaConf

__version__ = version(__package__ or __name__)

# arithmetic in interpolations, e.g. "${eval:(${episode_length} + ${action_horizon} - 1) * 10}i"
OmegaConf.register_new_resolver("eval", eval, replace=True)  # noqa: RUF067, S307

__all__ = ["__version__"]
