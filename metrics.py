import torch
import torch.nn as nn
from torchvision import models
from flow_utils import normalize_flow, warp
from einops import rearrange, repeat

class Metrics(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Choose RAFT to calculate flow for metrics
        if config.train.flow_model == 'raft':
            print('using RAFT')
            from flow_utils import RAFT
            self.flow_net = RAFT(model=config.train.flow_model_type, num_iters=config.train.raft_iters)
        
        # Choose GMFlow to calculate flow for metrics
        elif config.train.flow_model == 'gmflow':
            print('using GMFlow')
            from flow_utils import GMFlow
            self.flow_net = GMFlow(model=config.train.flow_model_type)
        
        # Choose PWCNet to calculate flow for metrics
        elif config.train.flow_model == 'pwcnet':
            print('using PWCNet')
            from flow_utils import PWC
            self.flow_net = PWC()
        
        else:
            raise NotImplementedError

        # Use L1 for metrics calculation
        self.criterion = nn.L1Loss()

    def compute_flow(self, im, frames, backward=False):
        '''
        Given an image (b, c, h, w) and frames (b, c, t, h, w) we compute 
            the flow between the image and all frames and return them
            in the shape (b, t, 2, h, w)
        If backward == True, then reverse the direction: compute flow from 
            all frames to single image
        '''
        b, c, t, h, w = frames.shape

        # Repeat and rearrange as needed
        im = repeat(im, 'b c h w -> (b num_frames) c h w', num_frames=t)
        frames = rearrange(frames, 'b c t h w -> (b t) c h w')

        # Compute flow
        if backward:
            flow = self.flow_net(frames, im)
        else:
            flow = self.flow_net(im, frames)

        # Rearrange to separate batch and time dim
        flow = rearrange(flow, '(b t) d h w -> b t d h w', b=b, t=t)

        return flow

    def compute_metrics(self, flow_pred, flow_src, alphas):
        # Calculate motion error in the way of mag_loss
        motion_error = self.criterion(flow_pred, alphas * flow_src)

        # Calculate motion error (ratio) between flow magnitudes ratio and alpha
        dist_pred = torch.sqrt(flow_pred[:,:,0]**2 + flow_pred[:,:,1]**2)
        dist_src = torch.sqrt(flow_src[:,:,0]**2 + flow_src[:,:,1]**2)
        mag_error = self.criterion(dist_pred / (dist_src + 0.0001), alphas)
        
        return motion_error, mag_error

    def forward(self, pred, src, alphas):
        '''
        pred: magnified future frame predictions (b, c, t-1, h, w)
        src: original frames (b, c, t, h, w)
        alphas: magnification factors (per batch)
        '''

        # Add dims to alpha to match flow (b, t, d, h, w)
        if len(alphas.shape) == 1:
            # Only batch dim, constant alpha per video
            alphas = alphas.view(-1,1,1,1,1)
        elif len(alphas.shape) == 3:
            # Spatially varying alpha, shape (b, h, w)
            b, h, w = alphas.shape
            alphas = alphas.view(b, 1, 1, h, w)

        # Get flows
        anchor_frame = src[:,:,0]
        flow_pred = self.compute_flow(anchor_frame, pred)
        flow_src = self.compute_flow(anchor_frame, src[:,:,1:])

        # Compute metrics
        motion_error, mag_error = self.compute_metrics(flow_pred, flow_src, alphas)

        return motion_error, mag_error
