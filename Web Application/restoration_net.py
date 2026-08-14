import torch.nn as nn
from models.backbone import RestorationBackboneLite
from models.ldmh import LocalDistributionMixtureHead, MixtureFiLMProjector
class BaselineNet(nn.Module):
    def __init__(self, base_ch=32, n_lr_blocks=4, n_hr_blocks=2, scale=2):
        super().__init__()
        self.backbone=RestorationBackboneLite(base_ch,n_lr_blocks,n_hr_blocks,scale)
    def forward(self,x): return self.backbone(x, None, None)
class DistributionMixtureRestorationNet(nn.Module):
    def __init__(self, base_ch=32, n_components=3, n_lr_blocks=4, n_hr_blocks=2, scale=2, use_film=True):
        super().__init__()
        self.use_film=use_film
        self.ldmh=LocalDistributionMixtureHead(in_ch=1, base_ch=base_ch, n_components=n_components)
        self.film_proj=MixtureFiLMProjector(n_components, base_ch)
        self.backbone=RestorationBackboneLite(base_ch,n_lr_blocks,n_hr_blocks,scale)
    def forward(self,x):
        mix_weights, beta, scale, _ = self.ldmh(x)
        if self.use_film:
            gamma, shift = self.film_proj(mix_weights)
        else:
            gamma, shift = None, None
        restored = self.backbone(x, gamma, shift)
        return restored, mix_weights, beta, scale
