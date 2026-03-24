from functools import wraps
from concurrent.futures import Executor
from rich.syntax import Syntax
from rich import print


def concurrent(f):

    @wraps(f)
    def _impl(executor: Executor, *args, **kwargs):
        task = executor.submit(f, *args, **kwargs)
        return task

    return _impl

def print_rich(text) -> None:
    text = str(text)
    syntax = Syntax(str(text), "yaml")
    print(syntax)