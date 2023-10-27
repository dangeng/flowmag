from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import TestTimeAdaptDataset
from myutils import AverageMeter

import matplotlib.pyplot as plt

def test_time_adapt(model, frames_dir, num_epochs=5, mode='first', device=0, inference_fn=None, inference_freq=1, alpha=None, save_dir=None, dataset_length=None):
    '''
    params:
        model: (nn.Module) model with checkpoint already loaded
        frames_dir: (string) path to directory of frames for test time adaptation (OPTIONAL NOW)
        dataset: optional dataset to give, default is TestTimeAdaptDataset
        num_epochs: (int) number of passes through the frames_dir
        mode: ['first', 'random'] how to sample frames
        device: device to put model and data on
        inference_fn: function to call at the end of each epoch

    output:
        model: (nn.Module) finetuned input module
    '''

    model.train()
    model = model.to(device)

    # Get dataset from frames
    # if dataset is None:
    dataset = TestTimeAdaptDataset(frames_dir, mode=mode, length=dataset_length)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=False)

    # Optimizer
    optimizer = Adam(model.module.trainable_parameters(), lr=1e-4)

    # Record losses
    meter_loss = AverageMeter('loss')
    meter_mag_loss = AverageMeter('loss_mag')
    meter_color_loss = AverageMeter('loss_color')

    # Record average of losses
    hist_loss = []
    hist_mag_loss = []
    hist_color_loss = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs}')

        # Run inference
        if inference_fn is not None and epoch % inference_freq == 0:
            print('Performing inference...')
            model.eval()
            inference_fn(model, epoch)
            model.train()

        # Save checkpoints of tta
        if save_dir is not None:
            print('saving epoch checkpoint...')
            chkpt = {'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
            torch.save(chkpt, save_dir / f'chkpt_{epoch:04}.pth')

        for cur_iter, frames in enumerate(tqdm(dataloader)):

            # Process data
            frames = frames.to(device)

            # Model
            preds, loss, info = model(frames, alpha=alpha)
            loss = loss.mean()

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track losses
            meter_loss.update(loss.item())
            meter_mag_loss.update(info['loss_mag'].item())
            meter_color_loss.update(info['loss_color'].item())

        print(f'Avg Loss: {meter_loss.avg}')
        print(f'Avg Mag Loss: {meter_mag_loss.avg}')
        print(f'Avg Color Loss: {meter_color_loss.avg}')
        hist_loss.append(meter_loss.avg)
        hist_mag_loss.append(meter_mag_loss.avg)
        hist_color_loss.append(meter_color_loss.avg)
        meter_loss.reset()
        meter_mag_loss.reset()
        meter_color_loss.reset()

        # Plot losses
        if save_dir is not None:
            loss_info = {'losses': hist_loss,
                        'mag_losses': hist_mag_loss,
                        'color_losses': hist_color_loss}
            for loss_name, losses in loss_info.items():
                plt.plot(losses)
                plt.title(loss_name)
                plt.savefig(save_dir / f'{loss_name}.png')
                plt.clf()

    # Final inference
    model.eval()
    if inference_fn is not None:
        inference_fn(model, epoch)

    info = {'losses': hist_loss,
            'mag_losses': hist_mag_loss,
            'color_losses': hist_color_loss}

    return model, info

