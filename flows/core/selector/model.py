from omegaconf import DictConfig
from typing import Union
from torch import nn
from flows.ml.models import create_model 


class ModelSelector:

    def select(config: DictConfig) -> nn.Module:
        """Flow only model is available, name=='flow'"""
        
        model = create_model(config.name, config.params)
        return model
