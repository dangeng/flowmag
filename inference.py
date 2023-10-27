import argparse
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.utils import save_image

from dataset import TrainingFramesDataset, FramesDataset
from test_time_adapt import test_time_adapt
from myutils import get_our_model, write_video, dist_transform


def inference(model, 
              frames_dataset, 
              save_dir, 
              alpha=2.0, 
              max_alpha=16.0, 
              mask=None, 
              num_device=1, 
              output_video=False):
    '''
    Takes frames_dataset, which should be ordered frames from a video
        and magnifies motion with respect to the first frame
        and saves them to the log file
    '''
    device = 'cuda'
    save_dir.mkdir(exist_ok=True, parents=True)
    results = []

    if isinstance(model, nn.Module):
        model.eval()
    
    training_status = model.module.get_training_status()

    # If input alpha exceeds the range for training, perform recursions for inference
    if alpha > max_alpha and np.sqrt(alpha) < max_alpha:
        our_alpha = np.sqrt(alpha)
        num_recursion = 2
    elif alpha < max_alpha:
        our_alpha = alpha
        num_recursion = 1
    else:
        raise Exception('alpha out of range')

    with torch.no_grad():
        im0 = frames_dataset[0][None].to(device)
        results.append(im0.detach().cpu())

        for i in tqdm(range(1, len(frames_dataset))):
            # Get i^th frame, and merge with 0^th frame
            im1 = frames_dataset[i][None].to(device)
            frames = torch.stack([im0, im1], dim=2).repeat(num_device,1,1,1,1)

            # Process frames
            for _ in range(num_recursion):
                if training_status:
                    pred, _, _ = model(frames, alpha=our_alpha, mask=mask)
                else:
                    pred = model(frames, alpha=our_alpha, mask=mask)
                frames = torch.stack([im0, pred[0,:,0].unsqueeze(0)], dim=2).repeat(num_device,1,1,1,1)

            # Save predicted frame
            pred = pred[0,:,0]
            results.append(pred.detach().cpu())

    # Save as video file
    if output_video:
        saved_frames = [(255*img.squeeze().permute(1,2,0).flip([-1]).numpy()).astype(np.uint8) for img in results]
        video_path = str(save_dir / f'x{alpha}.mp4') if mask is None else str(save_dir / f'masked_x{alpha}.mp4')
        write_video(saved_frames, 30, video_path)
        print('saved the video to {}'.format(video_path))
    
    # Save as image files
    else:
        save_dir = save_dir / f'x{alpha}' if mask is None else save_dir / f'masked_x{alpha}'
        save_dir.mkdir(exist_ok=True, parents=True)
        for i, img in enumerate(results):
            save_image(img, save_dir / f'{i+1:04}.png')
        print('saved the images to {}'.format(str(save_dir)))


if __name__ == '__main__':
    # Load args and config
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--frames_dir', type=str, required=True, help='path to directory of frames to magnify')
    parser.add_argument('--resume', type=str, help='path to checkpoint to resume from')
    parser.add_argument('--save_name', type=str, required=True, help='name to save under')
    parser.add_argument('--alpha', type=float, required=True, help='amount to magnify motion')
    parser.add_argument('--mask_path', type=str, default=None, help='path to numpy mask, if None then no mask')
    parser.add_argument('--soft_mask', type=int, default=0, help='how much to soften mask. 0 is none, higher is more')
    parser.add_argument('--output_video', action='store_true')
    parser.add_argument('--test_time_adapt', action='store_true')
    parser.add_argument('--tta_epoch', type=int, default=3, help='number of epochs for test time adaptation')
    args = parser.parse_args()

    # Make frames dataset
    frames_dataset = TrainingFramesDataset(args.frames_dir) if args.test_time_adapt else FramesDataset(args.frames_dir)

    # Load and preprocess mask, if given
    mask = None
    if args.mask_path:
        # Load mask from npy file
        mask = np.load(args.mask_path)
        mask = torch.tensor(mask)
        h, w = mask.shape
        mask = mask.float()

        if args.soft_mask:
            print('Softening mask')
            dist = dist_transform(mask)
            dist[dist < args.soft_mask] = 1
            dist[dist >= args.soft_mask] = 0
            mask = dist

    # Get max alpha
    config = OmegaConf.load(args.config)
    max_alpha = config.train.alpha_high

    # Make save dir
    save_dir = Path(args.resume).parent.parent / 'inference' / args.save_name
    save_dir.mkdir(exist_ok=True, parents=True)


    # Make model
    model, epoch = get_our_model(args, args.test_time_adapt)

    # Test time adaptation
    if args.test_time_adapt:
        save_dir = save_dir / f'tta_epoch{epoch:03}'
        save_dir.mkdir(exist_ok=True, parents=True)
        def inference_fn(model, epoch):
            new_save_dir = save_dir / f'tta_epoch{epoch:03}'
            new_save_dir.mkdir(exist_ok=True, parents=True)
            inference(model, 
                        frames_dataset, 
                        new_save_dir, 
                        alpha=args.alpha, 
                        max_alpha=max_alpha, 
                        mask=mask, 
                        num_device=1, 
                        output_video=args.output_video)

        # Run test time adaptation
        model, loss_info = test_time_adapt(model, args.frames_dir, num_epochs=args.tta_epoch, inference_fn=inference_fn, inference_freq=1, alpha=None, save_dir=save_dir, dataset_length=50000)

        # Save loss curve as images
        for loss_name, losses in loss_info.items():
            plt.plot(losses)
            plt.title(loss_name)
            plt.savefig(save_dir / f'{loss_name}.png')
            plt.clf()


    inference(model, frames_dataset, save_dir, alpha=args.alpha, max_alpha=max_alpha, mask=mask, num_device=1, output_video=args.output_video)
