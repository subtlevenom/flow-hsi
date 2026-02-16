import torch
from einops import einsum
from gpytorch.distributions import MultivariateNormal as Normal


class MultivariateNormal(Normal):

    def conditional_distribution(self, x:torch.Tensor, x_dims=[0,1,2]) -> 'MultivariateNormal':
        # 2. Condition on X = x
        x = x.permute(0,2,3,1)
        mean = super().mean
        cov = self.covariance_matrix

        y_dims = list(set(range(self.base_sample_shape[0])) - set(x_dims))

        mean_x = mean[...,x_dims]
        mean_y = mean[...,y_dims]

        Sxx = cov[...,x_dims,:][...,x_dims]
        Sxy = cov[...,x_dims,:][...,y_dims]
        Syx = cov[...,y_dims,:][...,x_dims]
        Syy = cov[...,y_dims,:][...,y_dims]
        Sxx_1 = torch.linalg.inv(Sxx)

        Q = torch.matmul(Syx, Sxx_1)
        # 3. Compute Conditional Parameters
        # mu1 + cov12 * inv(cov22) * (a - mu2)
        cond_mean = mean_y + torch.einsum('bijmn,bijn->bijm', Q, (x-mean_x))
        # cov11 - cov12 * inv(cov22) * cov21
        cond_cov = Syy - torch.matmul(Q, Sxy)

        return MultivariateNormal(mean=cond_mean, covariance_matrix=cond_cov)
    
    @property
    def mean(self) -> torch.Tensor:
        return super().mean.permute(0,3,1,2)

    def log_prob(self, x:torch.Tensor) -> torch.Tensor:
        return super().log_prob(x.permute(0,2,3,1))

    def prob(self, x:torch.Tensor) -> torch.Tensor:
        return torch.exp(super().log_prob(x.permute(0,2,3,1)))
