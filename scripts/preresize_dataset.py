import os
import cv2
import json
import tqdm
import shutil
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class ResizeSet(Dataset):
    def __init__(self, data_root, split, img_size=512):
        self.data_root = data_root

        self.transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
        self.frameA_root = os.path.join(self.data_root, split, 'frameA')
        self.frameB_root = os.path.join(self.data_root, split, 'frameB')

        with open(os.path.join(self.data_root, '{}_fn.json'.format(split)), 'r') as f:
            self.fn = json.load(f)

    def __getitem__(self, index):
        image_paths = [os.path.join(self.frameA_root, self.fn[index]), os.path.join(self.frameB_root, self.fn[index])]
        images = [self.transforms(Image.open(path)) for path in image_paths]
        return images, self.fn[index]

    def __len__(self):
        return len(self.fn)


def get_loader(data_root, split):
    dataset = ResizeSet(data_root, split)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

def resize_dataset(data_root, split, save_root):
    loader = ResizeSet(data_root, split)

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    if os.path.exists(os.path.join(save_root, split)):
        shutil.rmtree(os.path.join(save_root, split))

    os.makedirs(os.path.join(save_root, split))
    os.makedirs(os.path.join(save_root, split, 'frameA'))
    os.makedirs(os.path.join(save_root, split, 'frameB'))

    for images, fn in tqdm.tqdm(loader):
        cv2.imwrite(os.path.join(save_root, split, 'frameA', fn), 255*images[0].squeeze().permute(1,2,0).flip([-1]).numpy())
        cv2.imwrite(os.path.join(save_root, split, 'frameB', fn), 255*images[1].squeeze().permute(1,2,0).flip([-1]).numpy())

