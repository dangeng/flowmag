import torch.nn as nn
from .unet_parts import Up, Down, DoubleConv, OutConv


ACTIVATION_MAP = {'relu': nn.ReLU,
                  'tanh': nn.Tanh,
                  'sigmoid': nn.Sigmoid}

class MLP(nn.Module):
    def __init__(self, num_neurons, activation='relu', final_activation=None):
        '''
        num_neurons : list of num neurons per layer
        activation : one of ['relu', 'tanh', 'sigmoid']
        final_activation : same as `activation`, if None, then don't apply final activation
        '''
        assert len(num_neurons) > 1, 'Need more than one layer'
        assert activation in ACTIVATION_MAP, 'Invalid activation!'

        super(MLP, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(len(num_neurons) - 1):
            self.layers.append(nn.Linear(num_neurons[i], num_neurons[i+1]))

        self.activation = ACTIVATION_MAP[activation]()

        if final_activation is not None:
            self.final_activation = ACTIVATION_MAP[final_activation]()
        else:
            self.final_activation = None

    def forward(self, x):
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x)

            # Don't apply activation after last layer
            if layer_idx != len(self.layers) - 1:
                x = self.activation(x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

class UNet(nn.Module):
    ''' 
    Modified From: https://github.com/milesial/Pytorch-UNet

    n_channels - # input channels
    n_classes - # output channels
    num_layers - # downsampling layers == # upsampling layers 
                 (only downsamples num_layers - 1 times though
                 b/c one layer doesn't downsample)
    ndf - # of filters
    bilinear - If true use bilinear upsampling, else use transposed convs
    skip - if false, zero out skip connections
    return_bottleneck - if True, also return bottleneck
    '''
    def __init__(self, n_channels=3, 
                       n_classes=3, 
                       num_layers=5, 
                       ndf=64, 
                       bilinear=True, 
                       skip=True, 
                       return_bottleneck=False,
                       final_activation=None):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.num_layers = num_layers
        self.ndf = ndf
        self.skip = skip
        self.return_bottleneck = return_bottleneck
        if final_activation is not None:
            self.final_activation = ACTIVATION_MAP[final_activation]()
        else:
            self.final_activation = None
        bilinear_factor = 2 if bilinear else 1

        self.down_layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            mult = 2**layer_idx

            # First layer we just conv
            if layer_idx == 0:
                self.down_layers.append(DoubleConv(n_channels, ndf * mult))
            else:
                in_channels = ndf * mult // 2
                out_channels = ndf * mult

                # If last down layer
                if layer_idx == self.num_layers - 1:
                    out_channels = out_channels // bilinear_factor

                self.down_layers.append(Down(in_channels, out_channels))

        self.up_layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            mult = 2**(self.num_layers - layer_idx - 1)

            in_channels = ndf * mult
            out_channels = ndf * mult // 2 // bilinear_factor

            # Last layer we just conv
            if layer_idx == self.num_layers - 1:
                self.up_layers.append(OutConv(in_channels, n_classes))

            # Second to last layer doesn't have bilinear factor
            elif layer_idx == self.num_layers - 2:
                out_channels = out_channels * bilinear_factor
                self.up_layers.append(Up(in_channels, out_channels, bilinear))

            else:
                self.up_layers.append(Up(in_channels, out_channels, bilinear))

    def encode(self, x):
        # Keep track of skip connections
        skips = []
        for layer in self.down_layers:
            x = layer(x)
            skips.append(x)

        bottleneck = skips.pop()

        return bottleneck, skips

    def decode(self, x, skips):
        # We want to access to skips in reverse order, ignoring the latest one
        skips = skips[::-1]
        # Append a dummy entry to match length of `up_layers` for `zip`
        skips.append(None)

        for idx, (layer, skip) in enumerate(zip(self.up_layers, skips)):

            if idx != self.num_layers - 1:
                if not self.skip:
                    skip = skip * 0
                x = layer(x, skip)

            # If last layer, no skip
            else:
                logits = layer(x)

        if self.final_activation is not None:
            logits = self.final_activation(logits)

        return logits

    def forward(self, x):
        bottleneck, skips = self.encode(x)

        logits = self.decode(bottleneck, skips)

        if self.return_bottleneck:
            return logits, bottleneck
        else:
            return logits

'''
model = MLP([784, 512, 256, 128, 64, 32, 10])
summary(model, (4,784))
'''

'''
model = UNet(3,3, ndf=16, num_layers=6, skip=False, return_bottleneck=True)
summary(model, (4,3,256,256))
'''
