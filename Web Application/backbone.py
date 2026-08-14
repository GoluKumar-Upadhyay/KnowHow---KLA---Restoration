import torch, torch.nn as nn, torch.nn.functional as F
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1); return x1 * x2
class NAFBlockLite(nn.Module):
    def __init__(self, ch, expand=2):
        super().__init__()
        hidden = ch*expand
        self.norm1=nn.GroupNorm(1,ch); self.conv1=nn.Conv2d(ch,hidden,1)
        self.dwconv=nn.Conv2d(hidden,hidden,3,padding=1,groups=hidden)
        self.gate=SimpleGate(); self.se=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(hidden//2,hidden//2,1))
        self.conv2=nn.Conv2d(hidden//2,ch,1)
        self.norm2=nn.GroupNorm(1,ch); self.conv3=nn.Conv2d(ch,hidden,1)
        self.gate2=SimpleGate(); self.conv4=nn.Conv2d(hidden//2,ch,1)
        self.alpha1=nn.Parameter(torch.zeros(1,ch,1,1)); self.alpha2=nn.Parameter(torch.zeros(1,ch,1,1))
    def forward(self,x):
        y=self.norm1(x); y=self.conv1(y); y=self.dwconv(y); y=self.gate(y); y=y*self.se(y); y=self.conv2(y)
        x=x+self.alpha1*y
        y=self.norm2(x); y=self.conv3(y); y=self.gate2(y); y=self.conv4(y)
        x=x+self.alpha2*y
        return x
class RestorationBackboneLite(nn.Module):
    def __init__(self, base_ch=32, n_lr_blocks=4, n_hr_blocks=2, scale=2):
        super().__init__()
        self.scale=scale
        self.stem=nn.Conv2d(1,base_ch,3,padding=1)
        self.lr_blocks=nn.ModuleList([NAFBlockLite(base_ch) for _ in range(n_lr_blocks)])
        self.upsample=nn.Sequential(nn.Conv2d(base_ch,base_ch*scale*scale,3,padding=1), nn.PixelShuffle(scale))
        self.hr_blocks=nn.ModuleList([NAFBlockLite(base_ch) for _ in range(n_hr_blocks)])
        self.head=nn.Conv2d(base_ch,1,3,padding=1)
    def forward(self,x,film_gamma=None,film_shift=None):
        feat=self.stem(x)
        if film_gamma is not None: feat=feat*(1+film_gamma)+film_shift
        for b in self.lr_blocks: feat=b(feat)
        feat=self.upsample(feat)
        for b in self.hr_blocks: feat=b(feat)
        out=self.head(feat)
        x_up=F.interpolate(x,scale_factor=self.scale,mode="bilinear",align_corners=False)
        return x_up+out
