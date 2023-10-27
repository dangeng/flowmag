'''
Utilities for flow prediction, including:
    torch modules for pwcnet and raft flow
    image warping
    flow clipping
'''
import pathlib
import argparse

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class RAFT(nn.Module):
    def __init__(self, model='things', num_iters=5, dropout=0):
        super(RAFT, self).__init__()
        
        from flow_models.raft import raft

        if model == 'things':
            model = 'raft-things.pth'
        else:
            raise NotImplementedError

        # Get location of checkpoints
        raft_dir = pathlib.Path(__file__).parent.absolute()/'flow_models'/'raft'

        # Emulate arguments
        args = argparse.Namespace()
        args.model = raft_dir / model
        args.small = False
        args.mixed_precision = True
        args.alternate_corr = False
        args.dropout = dropout

        flowNet = nn.DataParallel(raft.RAFT(args))
        flowNet.load_state_dict(torch.load(args.model, map_location='cpu'))
        self.flowNet = flowNet.module.cpu()

        self.num_iters = num_iters

    def forward(self, im1, im2):
        '''
        Input: images \in [0,1]
        '''
        # Normalize to [0, 255]
        im1 = im1 * 255
        im2 = im2 * 255

        # Estimate flow
        flow_low, flow_up = self.flowNet(im1, im2, iters=self.num_iters, test_mode=True)

        return flow_up

class ARFlow(nn.Module):
    def __init__(self):
        super(ARFlow, self).__init__()
        
        from flow_models.ARFlow.models.pwclite import PWCLite
        from easydict import EasyDict
        from utils.torch_utils import restore_model

        chkpt_path = pathlib.Path(__file__).parent.absolute() / 'flow_models/ARFlow/checkpoints/KITTI15/pwclite_ar.tar'

        model_cfg = {
            'upsample': True,
            'n_frames': 2,
            'reduce_dense': True,
        }
        model_cfg = EasyDict(model_cfg)
        flowNet = PWCLite(model_cfg)
        flowNet = restore_model(flowNet, chkpt_path)
        self.flowNet = flowNet

    def forward(self, im1, im2):
        '''
        Input: images \in [0,1]
        '''
        inp = torch.cat([im1, im2], dim=1)
        return self.flowNet(inp)['flows_fw'][0]

class GMFlow(nn.Module):
    def __init__(self, model='things'):
        super(GMFlow, self).__init__()
        
        from flow_models.gmflow import gmflow
        if model == 'sintel':
            model = 'gmflow_sintel-0c07dcb3.pth'
        elif model == 'things':
            model = 'gmflow_things-e9887eda.pth'
        else:
            raise NotImplementedError

        # Get location of checkpoints
        gmflow_dir = pathlib.Path(__file__).parent.absolute()/'flow_models'/'gmflow'
        chkpt_path = gmflow_dir / model

        flowNet = gmflow.GMFlow(feature_channels=128,
                                                num_scales=1,
                                                upsample_factor=8,
                                                num_head=1,
                                                attention_type='swin',
                                                ffn_dim_expansion=4,
                                                num_transformer_layers=6,
                                                )
        checkpoint = torch.load(chkpt_path, map_location='cpu')
        flowNet.load_state_dict(checkpoint['model'])
        self.flowNet = flowNet

    def forward(self, im1, im2):
        '''
        Input: images \in [0,1]
        '''
        # Normalize to [0, 255]
        im1 = im1 * 255
        im2 = im2 * 255

        # Estimate flow
        results_dict = self.flowNet(im1, im2,
                                    attn_splits_list=[2],
                                    corr_radius_list=[-1],
                                    prop_radius_list=[-1],
                                    pred_bidir_flow=False
                                    )
        return results_dict['flow_preds'][-1]

class PWC(nn.Module):
    def __init__(self):
        super(PWC, self).__init__()

        from flow_models.pwcnet.pwc import Network
        self.flowNet = Network().eval().cpu()
        
    def forward(self, im1, im2):
        im1 = im1.squeeze()
        im2 = im2.squeeze()

        intWidth = im1.shape[2]
        intHeight = im1.shape[1]

        tenPreprocessedFirst = im1.cuda().view(1, 3, intHeight, intWidth)
        tenPreprocessedSecond = im2.cuda().view(1, 3, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

        tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

        tenFlow = 20.0 * torch.nn.functional.interpolate(input=self.flowNet(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tenFlow

def normalize_flow(flow):
    '''
    Normalize pixel-offset (relative) flow to absolute [-1, 1] flow
    input :
        flow : tensor (b, 2, h, w)
    output :
        flow : tensor (b, h, w, 2) (for `F.grid_sample`)
    '''
    _, _, h, w = flow.shape
    device = flow.device

    # Get base pixel coordinates (just "gaussian integers")
    base = torch.meshgrid(torch.arange(h), torch.arange(w))[::-1]
    base = torch.stack(base).float().to(device)

    # Convert to absolute coordinates
    flow = flow + base

    # Convert to [-1, 1] for grid_sample
    size = torch.tensor([w, h]).float().to(device)
    flow = -1 + 2.*flow/(-1 + size)[:,None,None]
    flow = flow.permute(0,2,3,1)
    
    return flow
    
def warp(im, flow, padding_mode='reflection'):
    '''
    requires absolute flow, normalized to [-1, 1]
        (see `normalize_flow` function)
    '''

    warped = F.grid_sample(im, flow, padding_mode=padding_mode, align_corners=True)

    return warped

if __name__=='__main__':
    flowNet = ARFlow()

    im1 = torch.randn(8,3,512,512)
    im2 = torch.randn(8,3,512,512)

    flow = flowNet(im1, im2)

    _ = 1