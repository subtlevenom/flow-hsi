from pathlib import Path

from ..ntire import datamodule as ntire
from .dataset import Dataset


def paired_paths(source, target):
    directories = (Path(source), Path(target))
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(f"Expected Volga2K image directory: {directory}.")
    images = [{p.name: str(p) for p in directory.iterdir()
               if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
              for directory in directories]
    sources, targets = images
    if not sources or sources.keys() != targets.keys():
        raise ValueError(
            f"Volga2K source/target filenames must match and be nonempty: "
            f"{source}, {target}; missing targets={sorted(sources.keys() - targets.keys())[:5]}, "
            f"missing sources={sorted(targets.keys() - sources.keys())[:5]}."
        )
    names = sorted(sources)
    return [sources[n] for n in names], [targets[n] for n in names]


class DataModule(ntire.DataModule):
    """Reuse seeded loaders and lazy split setup for paired Volga2K RGB images."""

    def _setup_split(self, split):
        if getattr(self, f"{split}_dataset") is not None:
            return
        config = self.splits[split]
        source, target = paired_paths(config["source"], config["target"])
        dataset = Dataset(source, target, crop_size=config.get("crop_size", 0),
                          random_crop=split == "train",
                          flip_probability=0.5 if split == "train" else 0.0,
                          filename=split == "predict")
        setattr(self, f"{split}_dataset", dataset)
