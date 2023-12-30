import os
import glob
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from subset_utils import motion_statistics, filtering_vis, get_relative_path

class VimeoTest(Dataset):
    def __init__(self, img_size, data_root, split):
        self.data_root = data_root

        self.transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        with open(os.path.join(data_root, 'vimeo/vimeo_septuplet/sep_{}list.txt'.format(split)), 'r') as file:
            self.folders = [os.path.join(data_root, 'vimeo/vimeo_septuplet/sequences', line.rstrip()) for line in file.readlines()]

        print('original dataset has {} folders'.format(len(self.folders)))

        self.nbr_frame = 2
        self.num_samples = len(self.folders) * (7 // self.nbr_frame)

    def __getitem__(self, index):
        # get frame pair sample given an index
        folder = self.folders[index // (7 // self.nbr_frame)]
        start_frame = (index % (7 // self.nbr_frame)) * self.nbr_frame
        paths = sorted(glob.glob(os.path.join(folder, '*')))[start_frame:start_frame+self.nbr_frame]
        images = []
        image_paths = []
        for path in paths:
            images.append(self.transforms(Image.open(path)))
            image_paths.append(get_relative_path(path, self.data_root))

        return images, image_paths

    def __len__(self):
        return self.num_samples


def get_loader(split, img_size, data_root, batch_size, shuffle, num_workers):
    dataset = VimeoTest(img_size, data_root, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--data_root', type=str, help='root folder to save data')
    args = parser.parse_args()
    data_root = args.data_root

    # calculate statistics for frame pairs
    loader = get_loader('train', img_size=512, data_root=data_root, batch_size=1, shuffle=False, num_workers=4)
    motion_statistics(data_root, loader, 'vimeo', 'train')

    loader = get_loader('test', img_size=512, data_root=data_root, batch_size=1, shuffle=False, num_workers=4)
    motion_statistics(data_root, loader, 'vimeo', 'test')

    
    # visualize with gif of frame pairs given a set of thresholds
    threshold_dict = {'max_dist': 20,
                      '80th': 2,
                      '0.01st': 0.1
                      }
    mse_threshold = 10
    flow_data_path = os.path.join(data_root, 'flow_data/vimeo_train.json')
    filtering_vis(data_root, 'vimeo', 'train', flow_data_path, threshold_dict, mse_threshold)

    flow_data_path = os.path.join(data_root, 'flow_data/vimeo_test.json')
    filtering_vis(data_root, 'vimeo', 'test', flow_data_path, threshold_dict, mse_threshold)
