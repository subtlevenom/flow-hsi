from typing import Optional
from . import cli


def main(cfg: dict) -> Optional[float]:
    entry = cli.register_task(cfg.get("task", None))
    return entry(cfg)


__all__ = ["main"]
