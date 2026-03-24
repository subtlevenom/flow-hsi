from pathlib import Path
from typing import Iterable


def files(path:Path | str, suffix:str=None) -> Iterable:
    """returns file iterator"""

    path:Path = Path(path)

    if not path.exists():
        print(f'Directory {path} does not exist')
        return 

    if path.is_file():
        if suffix is None or suffix == path.suffix:
            yield path 
    else:
        for f in path.glob('**/*' + (suffix or '')):
            if f.is_file():
                yield f

