import os
import glob
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from subset_utils import motion_statistics, filtering_vis, get_relative_path

class TaoTest(Dataset):
    def __init__(self, img_size, data_root, split, sampling=1):
        nbr_frame = 2
        self.data_root = data_root

        self.transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        all_folders = glob.glob(os.path.join(self.data_root, 'tao/frames', split, '*' ,'*'))
        print('original dataset has {} folders'.format(len(all_folders)))
        folders = []
        num_samples = [0]
        total_num = 0

        # record the accumulated num of samples from the first video
        for f in all_folders:
            if len(glob.glob(os.path.join(f, '*'))) >= nbr_frame + sampling - 1:
                folders.append(f)
                total_num += (len(glob.glob(os.path.join(f, '*'))) - sampling + 1) // nbr_frame
                num_samples.append(total_num)
        self.folders = folders
        self.nbr_frame = nbr_frame
        self.num_samples = num_samples
        self.sampling = sampling

    def __getitem__(self, index):
        # get frame pair sample given an index
        closest_value = min(self.num_samples, key=lambda x:abs(x-index))
        closest_index = self.num_samples.index(closest_value)
        if closest_value > index:
            folder = self.folders[closest_index - 1]
            start_frame = (index - self.num_samples[closest_index - 1]) * self.nbr_frame
        elif closest_value == index:
            folder = self.folders[closest_index]
            start_frame = 0
        else:
            folder = self.folders[closest_index]
            start_frame = (index - closest_value) * self.nbr_frame
        all_paths = sorted(glob.glob(os.path.join(folder, '*')))
        paths = [all_paths[start_frame], all_paths[start_frame+self.sampling]]
        images = []
        image_paths = []
        for path in paths:
            images.append(self.transforms(Image.open(path)))
            image_paths.append(get_relative_path(path, self.data_root))

        return images, image_paths

    def __len__(self):
        return self.num_samples[-1]


def get_loader(sampling, split, img_size, data_root, batch_size, shuffle, num_workers):
    dataset = TaoTest(img_size, data_root, split, sampling)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--data_root', type=str, help='root folder to save data')
    args = parser.parse_args()
    data_root = args.data_root
    
    # calculate statistics for frame pairs
    # two sampling methods (stride=1 or 5) are applied
    sampling = 1
    loader = get_loader(sampling, 'train', img_size=512, data_root=data_root, batch_size=1, shuffle=False, num_workers=4)
    motion_statistics(data_root, loader, 'tao', 'train_n{}'.format(sampling))

    sampling = 5
    loader = get_loader(sampling, 'train', img_size=512, data_root=data_root, batch_size=1, shuffle=False, num_workers=4)
    motion_statistics(data_root, loader, 'tao', 'train_n{}'.format(sampling))
    
    sampling = 1
    loader = get_loader(sampling, 'val', img_size=512, data_root=data_root, batch_size=1, shuffle=False, num_workers=4)
    motion_statistics(data_root, loader, 'tao', 'val_n{}'.format(sampling))

    sampling = 5
    loader = get_loader(sampling, 'val', img_size=512, data_root=data_root, batch_size=1, shuffle=False, num_workers=4)
    motion_statistics(data_root, loader, 'tao', 'val_n{}'.format(sampling))
    
    # visualize with gif of frame pairs given a set of thresholds
    threshold_dict = {'max_dist': 20,
                      '80th': 2,
                      '0.01st': 0.1
                      }
    mse_threshold = 10

    flow_data_path = os.path.join(data_root, 'flow_data/tao_train_n1.json')
    filtering_vis(data_root, 'tao', 'train_n1', flow_data_path, threshold_dict, mse_threshold)
    flow_data_path = os.path.join(data_root, 'flow_data/tao_train_n5.json')
    filtering_vis(data_root, 'tao', 'train_n5', flow_data_path, threshold_dict, mse_threshold)
    
    flow_data_path = os.path.join(data_root, 'flow_data/tao_val_n1.json')
    filtering_vis(data_root, 'tao', 'val_n1', flow_data_path, threshold_dict, mse_threshold)
    flow_data_path = os.path.join(data_root, 'flow_data/tao_val_n5.json')
    filtering_vis(data_root, 'tao', 'val_n5', flow_data_path, threshold_dict, mse_threshold)
