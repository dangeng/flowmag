import shutil
import sys
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path
import logging
from datetime import datetime

import torch
import os
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import MotionMagModel
from dataset import TrainingFramesDataset, RepeatDataset, get_dataloader
from myutils import AverageMeter, log_images
from inference import inference


# Load args and config
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--config', type=str, required=True, help='path to config file')
args = parser.parse_args()
config = OmegaConf.load(args.config)
config.config = args.config

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in range(config.train.ngpus)])

# Logging
config.loss_names = ['Loss', 'MagLoss', 'ColorLoss']

now_str = datetime.now().strftime('%m_%d_%Y-%H-%M-%S')

writer = SummaryWriter(log_dir=os.path.join(config.log.log_dir, f'{now_str}-{config.name}', 'runs'), comment=f'.{config.name}')
config.save_dir = Path(config.log.log_dir) / f'{now_str}-{config.name}'
config.save_dir.mkdir(exist_ok=True, parents=True)
logging.basicConfig(filename=config.save_dir / 'logs.txt', filemode='w', format='[%(asctime)s] %(message)s', level=logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(message)s')
stdout_handler.setFormatter(formatter)
logging.getLogger().addHandler(stdout_handler)
shutil.copyfile(args.config, config.save_dir / 'config.yaml')

# Log config
logging.info(config)

# Dataloaders
logging.info('Making dataloaders')
trainset = get_dataloader(config, 'train')
trainset = RepeatDataset(trainset, config.data.repeat_factor)
valset = get_dataloader(config, 'valid')
trainloader = DataLoader(trainset, batch_size=config.data.batch_size, shuffle=True, num_workers=config.data.num_workers, drop_last=True)
valloader = DataLoader(valset, batch_size=config.data.batch_size, shuffle=False, num_workers=4, drop_last=True)
frames_dataset = TrainingFramesDataset(config.log.inference_dir)

# Make models
logging.info('Making model')
device = 'cuda'
dp_devices = list(range(config.train.ngpus))
model = MotionMagModel(config).to(device)
model = nn.DataParallel(model)

# Optimizer
logging.info('Making optimizer')
optimizer = Adam(model.module.trainable_parameters(), lr=config.train.lr)

# Resume
if config.train.resume:
    logging.info(f'Resuming from {config.train.resume}')
    chkpt = torch.load(config.train.resume, map_location='cpu')
    model.load_state_dict(chkpt['state_dict'], strict=False)
    optimizer.load_state_dict(chkpt['optimizer_state_dict'])
    start_epoch = chkpt['epoch']
else:
    start_epoch = 0
if config.log.reset_epoch:
    start_epoch = 0


def train(model, trainloader, optimizer, config, writer, epoch_num):
    '''
    Train for one epoch
    '''
    metrics = {name: AverageMeter(name) for name in config.loss_names}
    model.train()

    for cur_iter, data in enumerate(tqdm(trainloader)):
        frames, info_data = data
        print(frames.shape)

        # Bookkeeping
        global_iter = (epoch_num - 1) * len(trainloader) + cur_iter

        # Process data
        frames = frames.to(device)

        # Model
        preds, loss, info = model(frames)
        loss = loss.mean()

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log
        metrics['Loss'].update(loss.item())
        metrics['MagLoss'].update(info['loss_mag'].mean().item())
        metrics['ColorLoss'].update(info['loss_color'].mean().item())
        if (cur_iter + 1) % config.log.print_freq == 0:
            log_strs = []
            for loss_name, meter in metrics.items():
                writer.add_scalar(f'{loss_name}/train', meter.avg, global_iter)
                log_strs.append(f'{loss_name} - {meter.avg:.6f}')
                meter.reset()
            log_str = f'[Epoch {epoch_num}] [Iter {cur_iter}/{len(trainloader)}] ' + ' | '.join(log_strs)
            logging.info(log_str)

    info = {
            'final_frames': frames,
            'final_preds': preds,
           }
    return info

def validate(model, valloader, config):
    '''
    Validate for one epoch
    '''
    metrics = {name: AverageMeter(name) for name in config.loss_names}
    model.eval()

    with torch.no_grad():
        for cur_iter, data in enumerate(tqdm(valloader)):
            frames, info_data = data

            # Process data
            frames = frames.to(device)

            # Model
            preds, loss, info = model(frames)
            loss = loss.mean()

            # Log metrics
            metrics['Loss'].update(loss.item())
            metrics['MagLoss'].update(info['loss_mag'].mean().item())
            metrics['ColorLoss'].update(info['loss_color'].mean().item())

    info_out = {
                'metrics': metrics,
                'final_frames': frames,
                'final_preds': preds,
            }

    return info_out

# Train loop
logging.info(f'Begin training')
for epoch_num in range(start_epoch + 1, config.train.num_epochs + 1):
    # Train
    logging.info(f'Training epoch {epoch_num}')
    train_info = train(model, trainloader, optimizer, config, writer, epoch_num)

    # Log images
    images_dict = {'frame': train_info['final_frames'],
                    'pred': train_info['final_preds']}
    log_images(writer, images_dict, epoch_num, 'train', config)

    # Validate
    if epoch_num % config.log.val_freq == 0:
        logging.info(f'Validating epoch {epoch_num}')
        val_info = validate(model, valloader, config)

        # Log val
        log_strs = []
        for loss_name, meter in val_info['metrics'].items():
            writer.add_scalar(f'{loss_name}/val', meter.avg, epoch_num)
            log_strs.append(f'{loss_name} - {meter.avg:.6f}')
            meter.reset()

        # Make imgs dict to save
        images_dict = {'frame': val_info['final_frames'],
                       'pred': val_info['final_preds']}
        if 'final_mags' in val_info:
            images_dict['mag'] = val_info['final_mags']

        log_images(writer, images_dict, epoch_num, 'val', config)
        log_str = f'[Epoch {epoch_num}] [Validation] ' + ' | '.join(log_strs)
        logging.info(log_str)

    # Inference
    if epoch_num % config.log.inference_freq == 0:
        logging.info(f'Inference on epoch {epoch_num}')
        save_dir = config.save_dir / 'inference' / f'epoch_{epoch_num:05}'
        inference(model, frames_dataset, save_dir, alpha=20, max_alpha=config.train.alpha_high, num_device=config.train.ngpus, output_video=True)

    # Save latest model
    logging.info(f'Saving epoch {epoch_num}')
    chkpt = {
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch_num,
            }
    save_path = config.save_dir / 'checkpoints'
    save_path.mkdir(exist_ok=True, parents=True)
    torch.save(chkpt, save_path / f'latest.pth')

    # Save model separately with a given save_freq
    if epoch_num % config.log.save_freq == 0:
        logging.info(f'Saving epoch {epoch_num}')
        chkpt = {
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch_num,
                }
        save_path = config.save_dir / 'checkpoints'
        save_path.mkdir(exist_ok=True, parents=True)
        torch.save(chkpt, save_path / f'chkpt_{epoch_num:05}.pth')