import torch 
import torch.nn as nn
import torch.nn.utils.spectral_norm as SpectralNorm

class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        ndf = params.filterD
        self.main = nn.Sequential(
            # input is (1) x 64 x 64
            SpectralNorm(nn.Conv2d(1, ndf, 4, stride=2, padding=1, bias=False)), 
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16 
            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )

    def forward(self, input):
        out = self.main(input)
        return torch.sum(out, [2, 3])
