from importlib.metadata import version

__version__ = version(__package__ or __name__)

__all__ = ["__version__"]
