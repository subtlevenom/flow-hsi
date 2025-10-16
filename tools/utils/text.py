import rich
from rich.syntax import Syntax


def print(code:str, lexer:str='yaml') -> None:
    syntax = Syntax(str(code), lexer)
    rich.print(syntax)