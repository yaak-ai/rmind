from omegaconf import OmegaConf


def register_custom_resolvers() -> None:
    if not OmegaConf.has_resolver("range"):
        OmegaConf.register_new_resolver("range", lambda start, stop: list(range(start, stop)))
