import os
import cv2
import sys
import tqdm
import json
import torch
import random
import shutil
import itertools
import numpy as np

sys.path.append('.')

from html_utils import HTML, save_gif
from flow_utils import RAFT


def get_relative_path(path, data_root):
    # remove root directory and get relative path
    return path.replace(os.path.abspath(data_root), '')[1:]

def get_absolute_path(path, data_root):
    # add root directory and get absolute path
    return os.path.join(data_root, path)

def motion_statistics(data_root, loader, dataset_name, split):
    # record the paths, mse, and percentiles of optical flow (motion) for frame pairs
    if not os.path.exists(os.path.join(data_root, 'flow_data')):
        os.mkdir(os.path.join(data_root, 'flow_data'))
    saved_path = os.path.join(data_root, 'flow_data/{}_{}.json'.format(dataset_name, split))
    print(dataset_name, split, saved_path)
    model = RAFT(model='things', num_iters=20)
    model.cuda()
    model.eval()

    flow_data = {}
    with torch.no_grad():
        for i, (images, image_paths) in enumerate(tqdm.tqdm(loader)):
            flow = model(images[0].cuda(), images[1].cuda())
            flow = np.abs(flow.squeeze().detach().cpu().numpy())
            flow_norm = np.sqrt(flow[0]**2 + flow[1]**2)
            paths = list(itertools.chain.from_iterable(image_paths))
            imgs = [cv2.imread(get_absolute_path(path, data_root)).astype(float) for path in paths]
            flow_data[i] = {'paths': tuple(sorted(paths)),
                            'mse': ((imgs[0] - imgs[1])**2).mean(),
                            'max_dist': np.percentile(flow_norm, 99.9),
                            '80th': np.percentile(flow_norm, 80),
                            '0.01st': np.percentile(flow_norm, 0.01),}
            
    with open(saved_path, 'w') as f:
        json.dump(flow_data, f, indent=4)


def filtering_vis(data_root, dataset_name, split, flow_data_path, threshold_dict, mse_threshold):
    # filter frame pairs with thresholds, and visualize the gif of frame pairs in html for inspection
    original_folders = []

    vis_path = os.path.join(data_root, '{}_{}_vis'.format(dataset_name, split))
    if os.path.exists(vis_path):
        shutil.rmtree(vis_path)
    os.makedirs(vis_path)
    
    with open(flow_data_path, 'r') as f:
        flow_data = json.load(f)
    included_data = {}
    filtered_data = {}
    for i, (_, value) in enumerate(tqdm.tqdm(flow_data.items())):
        image_paths = value['paths']
        if value['mse'] <= mse_threshold:
            save = False
        else:
            for key, threshold in threshold_dict.items():
                if value[key] > threshold:
                    save = False
                    break
                else:
                    save = True
            
        if save == False:
            filtered_data[i] = value

        else:
            included_data[i] = value
            original_folders.append(image_paths[0].split('/')[-2])


    print('number of the selected videos: {}/{}, from {} original folders'.format(len(included_data), len(filtered_data), len(set(original_folders))))

    with open(os.path.join(data_root, 'flow_data/{}_{}_filtered.json'.format(dataset_name, split)), 'w') as f:
        json.dump(filtered_data, f, indent=4)

    with open(os.path.join(data_root, 'flow_data/{}_{}_included.json'.format(dataset_name, split)), 'w') as f:
        json.dump(included_data, f, indent=4)

    print('saving html to {}'.format(os.path.join(data_root, '{}_{}_vis'.format(dataset_name, split))))
    html = HTML(os.path.join(data_root, '{}_{}_vis'.format(dataset_name, split)), 'subset')
    html.add_header(data_root.split('/')[-1])

    num_vis = 30
    filtered_vis = random.sample(list(filtered_data.keys()), num_vis)
    included_vis = random.sample(list(included_data.keys()), num_vis)

    html.add_header('dataset: {}, split: {}, threshold: {}, {} (MSE)'.format(dataset_name, split, str(threshold_dict), mse_threshold))
    html.add_header('filtered videos, {}/{}'.format(len(filtered_data), len(filtered_data)+len(included_data)))
    ims = []
    txts = []
    links = []
    for n in filtered_vis:
        paths = [get_absolute_path(path, data_root) for path in filtered_data[n]['paths']]
        gif_path = save_gif(paths, os.path.join(data_root, '{}_{}_vis'.format(dataset_name, split), 'images'))
        ims.append(gif_path)
        txts.append(str(filtered_data[n]))
        links.append(gif_path)
    html.add_images(ims, txts, links)

    html.add_header('included videos, {}/{}'.format(len(included_data), len(filtered_data)+len(included_data)))
    ims = []
    txts = []
    links = []
    for n in included_vis:
        paths = [get_absolute_path(path, data_root) for path in included_data[n]['paths']]
        gif_path = save_gif(paths, os.path.join(data_root, '{}_{}_vis'.format(dataset_name, split), 'images'))
        ims.append(gif_path)
        txts.append(str(included_data[n]))
        links.append(gif_path)
    html.add_images(ims, txts, links)
    html.save()