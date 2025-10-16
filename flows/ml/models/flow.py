from typing import Dict, List
from torch import nn
import graphkit as gk


class Flow(nn.Module):

    def __init__(self, modules: Dict[str, nn.Module], metadata: List):
        super(Flow, self).__init__()
        self.layers = nn.ModuleDict(modules)
        self.graph = self.compose(metadata)

    def compose(self, metadata):
        ops = [
            gk.operation(
                name=f'{i}.{m.layer}', needs=list(m.inputs),
                provides=list(m.outputs))(self.layers[m.layer])
            for i, m in enumerate(metadata)
        ]
        return gk.compose(name='Flow')(*ops)

    def forward(self, **kwargs):
        return self.graph(kwargs)
