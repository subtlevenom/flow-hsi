from .logger import Logger

__all__ = ["Logger", "DataSelector", "ModelSelector", "PipelineSelector"]


def __getattr__(name):
    # Spawned dataset workers import Logger before selectors/datasets finish
    # loading. Resolve selectors on demand to avoid re-entering that import.
    if name in ("DataSelector", "ModelSelector", "PipelineSelector"):
        from . import selector
        return getattr(selector, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")