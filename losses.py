import torch.nn as nn
from flow_utils import normalize_flow, warp
from einops import rearrange, repeat

class MMLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Choose RAFT to calculate flow
        if config.train.flow_model == 'raft':
            from flow_utils import RAFT
            self.flow_net = RAFT(num_iters=config.train.raft_iters)
        
        # Choose ARFlow to calculate flow
        elif config.train.flow_model == 'arflow':
            from flow_utils import ARFlow
            self.flow_net = ARFlow()

        else:
            raise NotImplementedError
        
        # Use L1 for loss calculation
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

    def compute_mag_loss(self, flow_pred, flow_src, alphas):
        # Enforce predicted flow == alpha * src flow
        return self.criterion(flow_pred, alphas * flow_src)

    def warp_frames(self, frames, flow):
        # Combine batch and time dims
        flow = rearrange(flow, 'b t d h w -> (b t) d h w')

        # Normalize flow
        flow = normalize_flow(flow)

        # Rearrange frames to match flows
        frames = rearrange(frames, 'b c t h w -> (b t) c h w')

        # Warp future frames into first
        frames_warped = warp(frames, flow)

        return frames_warped


    def compute_color_loss(self, pred, src, flow_pred, flow_src):
        info = {}

        # Warp frames into first
        src_warped = self.warp_frames(src[:,:,1:], flow_src)
        pred_warped = self.warp_frames(pred, flow_pred)

        # Enforce the pixels in pred videos keep consistent with the input video
        color_loss = self.criterion(pred_warped, src_warped)

        # Add warps to info
        info['pred_warped'] = pred_warped
        info['src_warped'] = src_warped

        return color_loss, info


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

        # Compute losses
        mag_loss = self.compute_mag_loss(flow_pred, flow_src, alphas)
        color_loss, info = self.compute_color_loss(pred, src, flow_pred, flow_src)

        # Add to info
        info['flow_pred'] = flow_pred
        info['flow_src'] = flow_src

        return mag_loss, color_loss, info
