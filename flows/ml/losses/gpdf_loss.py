import torch
from torch import nn
from torchmetrics import Metric
import torch.nn.functional as F
import numpy as np
from flows.ml.layers.sep_gpd import MultivariateNormal
import einops


class GPDFLoss(Metric):
    """
    gpdf_loss = kl(q||p) + kl(p||q) - mx,my correlation. So we ignore (mx-my)*S^-1*(mx-my) term
    https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
    https://arxiv.org/pdf/2102.05485
    """

    def __init__(self):
        super(GPDFLoss, self).__init__()
        self.add_state("correct",
                       default=torch.tensor(0, dtype=torch.float64),
                       dist_reduce_fx="sum")
        self.add_state("total",
                       default=torch.tensor(0, dtype=torch.long),
                       dist_reduce_fx="sum")

    def update(
        self,
        q: MultivariateNormal,
        p: MultivariateNormal,
    ) -> None:
        Sq = q.covariance_matrix
        Sp = p.covariance_matrix

        Sq_1 = q.precision_matrix
        Sp_1 = p.precision_matrix

        Sq_1_Sp = torch.einsum('bhwij,bhwjk->bhwik', Sq_1, Sp)
        Sp_1_Sq = torch.einsum('bhwij,bhwjk->bhwik', Sp_1, Sq)

        trace_Sq_1_Sp = torch.einsum("...ii", Sq_1_Sp)
        trace_Sp_1_Sq = torch.einsum("...ii", Sp_1_Sq)

        n = Sq.shape[-1]

        d = 0.5 * (trace_Sq_1_Sp + trace_Sp_1_Sq) - n
        loss = torch.mean(d)

        self.correct += loss
        self.total += 1

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total
