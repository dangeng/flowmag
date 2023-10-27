from omegaconf import OmegaConf
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from einops import rearrange
from model import MotionMagModel

class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.data = []

    def update(self, val, n=1):
        # Compute the sum, avg, std and standard error for data
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.data.append(val)
        self.std = np.std(self.data)
        self.se = np.std(self.data, ddof=1) / np.sqrt(np.size(self.data))

    def __str__(self):
        return f"{self.name}: {self.val:.5f} {self.avg:.5f}"


def write_video(frames, fps, output_path):
    # Write a list of array for frames into mp4 files
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w,h))
    for frame in frames:
        writer.write(frame)

    writer.release()


def get_our_model(args, training=False):
    # Get our model for inference or evaluation
    config = OmegaConf.load(args.config)
    config.config = args.config

    # Force configs
    config.train.ngpus = 1
    config.train.is_training = True if training else False
    config.data.batch_size = 1

    # Make model
    print('Making model')
    device = 0
    dp_devices = list(range(1))
    model = MotionMagModel(config).to(device)
    model = nn.DataParallel(model, device_ids=dp_devices)

    # Resume model
    print(f'Resuming from {args.resume}')
    chkpt = torch.load(args.resume)
    model.load_state_dict(chkpt['state_dict'], strict=False)

    return model, chkpt['epoch']


def dist_transform(mask):
    h, w = mask.shape

    # get closest in south east
    def closest_se(mask):
        closest = torch.ones(mask.shape) * float('inf')
        for i in tqdm(range(1, h)):
            for j in range(1, w):
                if mask[i,j] == 1:
                    closest[i, j] = 0
                else:
                    closest[i, j] = min(closest[i,j-1], closest[i-1,j]) + 1

        return closest

    # Get all four possible directions
    se = closest_se(mask)
    nw = closest_se(mask.flip((0,1))).flip((0,1))
    sw = closest_se(mask.flip(0)).flip(0)
    ne = closest_se(mask.flip(1)).flip(1)

    res_0 = torch.minimum(se, nw)
    res_1 = torch.minimum(sw, ne)
    res = torch.minimum(res_0, res_1)

    return res


def log_images(writer, images_dict, epoch_num, split, config):
    # save images
    save_dir = config.save_dir / 'images'
    save_dir.mkdir(exist_ok=True, parents=True)

    for name, images in images_dict.items():
        for idx, image in enumerate(rearrange(images, 'b c t h w -> t b c h w')):
            grid = make_grid(image)
            writer.add_image(f'{name}_{idx}/{split}', grid, epoch_num)
            save_image(grid, save_dir / f'epoch{epoch_num:05}.{split}.{name}{idx:02}.png')