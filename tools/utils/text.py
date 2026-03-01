import json
from omegaconf import DictConfig, OmegaConf
import rich
import rich.json
from rich.syntax import Syntax
from rich.markdown import Markdown
import yaml


def rprint(code:str, lexer:str='json') -> None:
    syntax = Syntax(code, lexer, line_numbers=True)
    rich.print(syntax)

def print_config(config:DictConfig) -> None:
    rprint(OmegaConf.to_yaml(config), 'yaml')

def print_json(data:dict) -> None:
    rprint(yaml.dump(data, sort_keys=False), 'yaml')

