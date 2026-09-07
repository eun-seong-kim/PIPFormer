import torch
from torch import nn
import numpy as np
from networks.basic_module import FullyConnectedLayer, Conv2dLayer, MinibatchStdLayer, DisFromRGB, DisBlock

def nf(stage):
    NF = {1024: 64, 512: 64, 256: 128, 128: 256, 64: 512, 32: 512, 16: 512, 8: 512, 4: 512}
    return NF[2 ** stage]


class Discriminator(torch.nn.Module):
    def __init__(self,
                 img_resolution,               # Input resolution.
                 img_channels,                 # Number of input color channels.
                 activation         = 'lrelu',
                 mbstd_group_size   = None,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
                 mbstd_num_channels = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
                 divide = 1
                 ):
        
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2 ** resolution_log2 and img_resolution >= 4
        self.resolution_log2 = resolution_log2


        Dis = [DisFromRGB(img_channels, nf(resolution_log2) // divide, activation)]
        for res in range(resolution_log2, 2, -1):
            Dis.append(DisBlock(nf(res) // divide, nf(res-1) // divide, activation))

        if mbstd_num_channels > 0:
            Dis.append(MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels))
        Dis.append(Conv2dLayer(nf(2) // divide + mbstd_num_channels, nf(2) // divide, kernel_size=3, activation=activation))
        self.Dis = nn.Sequential(*Dis)

        self.fc0 = FullyConnectedLayer(nf(2)*4**2 // divide, nf(2) // divide, activation=activation)
        self.fc1 = FullyConnectedLayer(nf(2) // divide, 1)
        
    
    def forward(self, images_in):
        x = self.Dis(images_in)
        out = self.fc1(self.fc0(x.flatten(start_dim=1)))
        return out