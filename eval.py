'''
Given a test set, generates and saves predictions to a folder
Actual evaluation is done by another script
'''
import argparse
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import get_dataloader
from metrics import Metrics
from myutils import AverageMeter, get_our_model
    


def test(args, model, save_dir):
    device = 0
    num_recursion = 2

    # Make place to save
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_file = save_dir / f'flowmag_{args.flow_model}_{args.flow_model_type}.txt'
    save_file.touch()
    print(f'will be saved to {str(save_file)}')

    # Make frames dataset
    config = OmegaConf.load(args.config)
    config.config = args.config
    dataset = get_dataloader(config, 'test')
    testloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Make loss function
    dotlist = [f'train.flow_model={args.flow_model}', f'train.flow_model_type={args.flow_model_type}', 'train.raft_iters=20'] 
    loss_conf = OmegaConf.from_dotlist(dotlist)
    metric = Metrics(loss_conf).to(device)
    alphas = torch.tensor(args.alpha).to(device).unsqueeze(0)

    model.eval()

    print(f'Evaluating...')
    Motion_error = AverageMeter('motion_error')
    Mag_error = AverageMeter('mag_error')
    with torch.no_grad():
        for idx, (frames, info_data) in enumerate(tqdm(testloader)):
            
            config = OmegaConf.load(args.config)

            frames = frames.to(device)
            
            # If input alpha exceeds the range for training, perform recursion for inference
            if args.alpha > config.train.alpha_high and np.sqrt(args.alpha) <= config.train.alpha_high:
                our_alpha = np.sqrt(args.alpha)
                frames_ = frames.clone()
                for _ in range(num_recursion):
                    pred = model(frames_, alpha=our_alpha)
                    frames_ = torch.stack([frames[:,:,0], pred[:,:,0]], dim=2).to(device)
            
            # Run inference, if alpha is within the range of alpha for training
            elif args.alpha <= config.train.alpha_high:
                pred = model(frames, alpha=args.alpha)
            
            # Alpha out of range with given num of recursion
            else:
                raise Exception('alpha out of range')

            # Get metrics
            motion_error, mag_error = metric(pred, frames, alphas)

            Motion_error.update(motion_error.item())
            Mag_error.update(mag_error.item())
            
    print(f'Alpha: {args.alpha}')
    print(f'Avg motion error: {Motion_error.avg:.04f}, std: {Motion_error.std:.04f}, standard error: {Motion_error.se:.04f} @ alpha={args.alpha}')
    print(f'Avg mag error: {Mag_error.avg:.04f}, std: {Mag_error.std:.04f}, standard error: {Mag_error.se:.04f} @ alpha={args.alpha}')

    # Write in a txt file
    with open(save_file, 'a+') as f:
        f.write(f'Alpha: {args.alpha}\n')
        f.write(f'\tAvg motion error: {Motion_error.avg:.04f}, std: {Motion_error.std:.04f}, standard error: {Motion_error.se:.04f}\n')
        f.write(f'\tAvg mag error: {Mag_error.avg:.04f}, std: {Mag_error.std:.04f}, standard error: {Mag_error.se:.04f}\n')

if __name__ == '__main__':

    # Load args and config
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--resume', type=str, help='path to checkpoint to resume from')
    parser.add_argument('--alpha', type=float, required=True, help='amount to magnify motion')
    parser.add_argument('--flow_model', type=str, default='raft')
    parser.add_argument('--flow_model_type', type=str, default='things')
    args = parser.parse_args()

    model, epoch = get_our_model(args)
    save_name = f"epoch{epoch:04}"
    config_name = args.config.split('/')[-1].split('.yaml')[-2]
    save_dir = f'./eval_results/{config_name}.ep{epoch}'

    test(args, model, save_dir)