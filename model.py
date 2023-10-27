import numpy as np
import torch
import torch.nn as nn
from einops import repeat, rearrange

from models.model_zoo import UNet
from losses import MMLoss

class MotionMagModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.generator = UNet(n_channels=config.data.num_frames * 3 + config.model.pos_dim, 
                              n_classes=(config.data.num_frames - 1) * 3, 
                              num_layers=config.model.num_layers, 
                              ndf=config.model.ndf, 
                              final_activation='sigmoid')
        self.alpha_appender = AlphaAppender(config)
        if config.train.is_training:
            self.training = True
            self.mm_loss_fn = MMLoss(config)
        else:
            self.training = False

    def get_training_status(self):
        return True if self.training else False
    
    def trainable_parameters(self):
        return self.generator.parameters()

    def forward(self, frames, alpha=None, mask=None):
        b, c, t, h, w = frames.shape

        # Predict
        frames_flat = rearrange(frames, 'b c t h w -> b (c t) h w')
        frames_flat, alphas = self.alpha_appender(frames_flat, alpha=alpha, mask=mask)
        preds = self.generator(frames_flat)
        preds = rearrange(preds, 'b (c t) h w -> b c t h w', c=c, t=t-1)

        # Compute Losses
        if self.training:
            loss_mag, loss_color, info = self.mm_loss_fn(preds, frames, alphas)
            loss_mag = loss_mag.mean()
            loss_color = loss_color.mean()
            loss = self.config.train.weight_mag * loss_mag + self.config.train.weight_color * loss_color

            # Add stuff to info
            info['loss_mag'] = loss_mag
            info['loss_color'] = loss_color

            return preds, loss, info
        
        else:
            return preds

class AlphaAppender(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha_low = config.train.alpha_low
        self.alpha_high = config.train.alpha_high
        self.pos_dim = config.model.pos_dim

    def sample_alphas(self, b):
        # Sample alpha exponentially with a size of b
        exp_low = np.log2(self.alpha_low)
        exp_high = np.log2(self.alpha_high)
        exps = np.random.uniform(exp_low, exp_high, b)
        alphas = 2**exps
        return torch.tensor(alphas).float()

    def get_positional_encoding(self, alphas):
        '''
        Given alphas of shape (batch_dim,), return positional encoding
            of shape (batch_dim, pos_dim)
        '''
        device = alphas.device

        # Wavelengths are geometric between 0.125 and 128
        wavelengths = torch.logspace(-3, 7, self.pos_dim, base=2).to(device)
        vals = torch.sin(2 * 3.141592 * alphas[:, None] / wavelengths[None, :])
        return vals

    def forward(self, frames, alpha=None, mask=None):
        b, c, h, w = frames.shape
        device = frames.device

        # Sample alphas
        if alpha is None:
            alphas = self.sample_alphas(b).to(device)
        else:
            alphas = torch.ones(b).to(device) * alpha

        # Get positional encoding
        pe = self.get_positional_encoding(alphas)
        pe = pe.to(frames.device)

        # Repeat spatially and concat
        pe = repeat(pe, 'b d -> b d h w', h=h, w=w)

        if mask is not None:
            # Add batch and pos_dim dims
            mask = mask[None, None]

            # Get "zero" motion mag
            zero_alpha = self.get_positional_encoding(torch.ones(b))
            zero_alpha = zero_alpha.to(frames.device)
            zero_alpha = repeat(zero_alpha, 'b d -> b d h w', h=h, w=w)

            # Replace zeros with "zero_alpha"
            pe = pe * mask + zero_alpha * (1 - mask)

        # Concatenate frames with alpha embedding
        frames_pe = torch.cat([frames, pe], dim=1)

        # Return both frames_pe and alphas used
        return frames_pe, alphas.to(frames.device)