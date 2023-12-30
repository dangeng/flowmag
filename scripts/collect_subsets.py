import os
import tqdm
import json
import random
import shutil
import argparse
from html_utils import HTML, save_gif
from subset_utils import get_absolute_path
from preresize_dataset import resize_dataset

# specify the root directory for data
parser = argparse.ArgumentParser(description='Process some inputs.')
parser.add_argument('--data_root', type=str, help='root folder to save data')
args = parser.parse_args()
data_root = args.data_root

def generate_trainset(info_path_dict, save_root, threshold_dict, mse_threshold):
    split = 'train'

    # create dataset folder
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    # delete the folder of split (if any) and create new folders
    if os.path.exists(os.path.join(save_root, split)):
        shutil.rmtree(os.path.join(save_root, split))
    
    os.makedirs(os.path.join(save_root, split))
    os.makedirs(os.path.join(save_root, split, 'frameA'))
    os.makedirs(os.path.join(save_root, split, 'frameB'))

    included_data = {}
    filtered_data = {}
    fn_dict = {}
    all_fn = []
    num_videos = 0

    for dataset, info_path in info_path_dict.items():
        fn_dict[dataset] = {}
        num = 0

        # read motion data from json file
        with open(info_path, 'r') as f:
            flow_data = json.load(f)

        for idx, value in tqdm.tqdm(flow_data.items()):

            # try threshold for mse
            if value['mse'] <= mse_threshold:
                save = False
            else:
                # try percentile thresholds for motion
                for key, threshold in threshold_dict.items():
                    if value[key] > threshold:
                        save = False
                        break
                    else:
                        save = True
            
            if save == False:
                # save info of filtered data
                filtered_data['{}_{}'.format(dataset, str(idx))] = value

            else:
                # save info of included data and copy image files to new directory
                included_data['{}_{}'.format(dataset, str(idx))] = value
                image_paths = value['paths']
                image_fn = '{}_{}.png'.format(dataset, str(num).zfill(5))
                fn_dict[dataset][idx] = {'fn': image_fn, 'frameA': image_paths[0], 'frameB': image_paths[1]}
                all_fn.append(image_fn)
                shutil.copy(get_absolute_path(image_paths[0], data_root), os.path.join(save_root, split, 'frameA', image_fn))
                shutil.copy(get_absolute_path(image_paths[1], data_root), os.path.join(save_root, split, 'frameB', image_fn))
                num += 1

        print('num of videos from {}: {}'.format(dataset, num))
        num_videos += num

    print('number of the selected videos: {}'.format(num_videos))

    # write the info and filenames of filtered/included data into json files
    with open(os.path.join(save_root, '{}_files.json'.format(split)), 'w') as f:
        json.dump(fn_dict, f, indent=4)

    with open(os.path.join(save_root, '{}_fn.json'.format(split)), 'w') as f:
        json.dump(all_fn, f, indent=4)
        
    with open(os.path.join(save_root, '{}_filtered.json'.format(split)), 'w') as f:
        json.dump(filtered_data, f, indent=4)

    with open(os.path.join(save_root, '{}_included.json'.format(split)), 'w') as f:
        json.dump(included_data, f, indent=4)


    # visualize the gif of frame pairs of filtered/included data
    print('saving html to {}'.format(os.path.join(save_root, '{}_vis'.format(split))))
    html = HTML(os.path.join(save_root, '{}_vis'.format(split)), 'subset')
    html.add_header(save_root.split('/')[-1])

    num_vis = 30
    filtered_vis = random.sample(list(filtered_data.keys()), num_vis)
    included_vis = random.sample(list(included_data.keys()), num_vis)

    html.add_header('split: {}, threshold: {}, {} (MSE)'.format(split, str(threshold_dict), str(mse_threshold)))
    html.add_header('filtered videos, {}/{}'.format(len(filtered_data), len(filtered_data)+len(included_data)))
    ims = []
    txts = []
    links = []
    for n in filtered_vis:
        paths = filtered_data[n]['paths']
        gif_path = save_gif(paths, os.path.join(save_root, '{}_vis'.format(split), 'images'))
        ims.append(gif_path)
        txts.append(str(filtered_data[n]))
        links.append(gif_path)
    html.add_images(ims, txts, links)

    html.add_header('included videos, {}/{}'.format(len(included_data), len(filtered_data)+len(included_data)))
    ims = []
    txts = []
    links = []
    for n in included_vis:
        paths = included_data[n]['paths']
        gif_path = save_gif(paths, os.path.join(save_root, '{}_vis'.format(split), 'images'))
        ims.append(gif_path)
        txts.append(str(included_data[n]))
        links.append(gif_path)
    html.add_images(ims, txts, links)
    html.save()


def generate_testset(info_path_dict, save_root, threshold_dict, mse_threshold, num_per_dataset=100):
    split = 'test'

    # create dataset folder
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    # delete the folder of split (if any) and create new folders
    if os.path.exists(os.path.join(save_root, split)):
        shutil.rmtree(os.path.join(save_root, split))
    
    os.makedirs(os.path.join(save_root, split))
    os.makedirs(os.path.join(save_root, split, 'frameA'))
    os.makedirs(os.path.join(save_root, split, 'frameB'))

    included_data = {}
    filtered_data = {}
    fn_dict = {}
    all_fn = []

    for dataset, info_path in info_path_dict.items():

        if dataset in ['tao_n1', 'tao_n5']:
            num_per_dataset_ = int(num_per_dataset / 2)

            # collect from every subsets for TAO dataset
            temp_filtered_data = {'ArgoVerse': {}, 'BDD': {}, 'Charades': {}, 'LaSOT': {}, 'YFCC100M': {}}
            temp_included_data = {'ArgoVerse': {}, 'BDD': {}, 'Charades': {}, 'LaSOT': {}, 'YFCC100M': {}}

            with open(info_path, 'r') as f:
                flow_data = json.load(f)


            for idx, value in tqdm.tqdm(flow_data.items()):
                image_paths = value['paths']
                sub_name = image_paths[0].split('/')[-3]

                # try threshold for mse
                if value['mse'] <= mse_threshold:
                    save = False
                else:
                    # try percentile thresholds for motion
                    for key, threshold in threshold_dict.items():
                        if value[key] > threshold:
                            save = False
                            break
                        else:
                            save = True

                if save == False:
                    # save info of filtered data
                    temp_filtered_data[sub_name][idx] = value
                    filtered_data['{}_{}'.format(dataset, str(idx))] = value

                else:
                    # save info of included data
                    temp_included_data[sub_name][idx] = value
                    included_data['{}_{}'.format(dataset, str(idx))] = value
        
            for key, value in temp_included_data.items():
                # randomly select from included data, save info and copy images to new folder
                fn_dict[dataset+'_'+key] = {}
                all_included = list(value.keys())
                random.shuffle(all_included)
                for i, idx in enumerate(all_included[:num_per_dataset_]):
                    image_paths = value[idx]['paths']
                    image_fn = '{}_{}_{}.png'.format(dataset, key, str(i).zfill(5))
                    fn_dict[dataset+'_'+key][i] = {'fn': image_fn, 'frameA': image_paths[0], 'frameB': image_paths[1]}
                    all_fn.append(image_fn)
                    shutil.copy(get_absolute_path(image_paths[0], data_root), os.path.join(save_root, split, 'frameA', image_fn))
                    shutil.copy(get_absolute_path(image_paths[1], data_root), os.path.join(save_root, split, 'frameB', image_fn))

        elif dataset in ['youtube_vos', 'vimeo', 'davis', 'uvo_n1', 'uvo_n5']:
            num_per_dataset_ = int(num_per_dataset / 2) if 'uvo' in dataset else num_per_dataset
            fn_dict[dataset] = {}

            with open(info_path, 'r') as f:
                flow_data = json.load(f)
            for idx, value in tqdm.tqdm(flow_data.items()):
                # try threshold for mse
                if value['mse'] <= mse_threshold:
                    save = False
                else:
                    # try percentile thresholds for motion
                    for key, threshold in threshold_dict.items():
                        if value[key] > threshold:
                            save = False
                            break
                        else:
                            save = True
                if save == False:
                    # save info of filtered data
                    filtered_data['{}_{}'.format(dataset, str(idx))] = value

                else:
                    # save info of included data
                    included_data['{}_{}'.format(dataset, str(idx))] = value
            
            # randomly select from included data, save info and copy images to new folder
            all_included = list(included_data.keys())
            random.shuffle(all_included)
            print(dataset)
            for i, idx in enumerate(all_included[:num_per_dataset_]):
                image_paths = included_data[idx]['paths']
                image_fn = '{}_{}.png'.format(dataset, str(i).zfill(5))
                fn_dict[dataset][i] = {'fn': image_fn, 'frameA': image_paths[0], 'frameB': image_paths[1]}
                all_fn.append(image_fn)
                shutil.copy(get_absolute_path(image_paths[0], data_root), os.path.join(save_root, split, 'frameA', image_fn))
                shutil.copy(get_absolute_path(image_paths[1], data_root), os.path.join(save_root, split, 'frameB', image_fn))

    print(len(all_fn))

    # save info and filenames
    with open(os.path.join(save_root, '{}_files.json'.format(split)), 'w') as f:
        json.dump(fn_dict, f, indent=4)

    with open(os.path.join(save_root, '{}_fn.json'.format(split)), 'w') as f:
        json.dump(all_fn, f, indent=4)
        
    with open(os.path.join(save_root, '{}_filtered.json'.format(split)), 'w') as f:
        json.dump(filtered_data, f, indent=4)

    with open(os.path.join(save_root, '{}_included.json'.format(split)), 'w') as f:
        json.dump(included_data, f, indent=4)


    # visualize with gif of frame pairs of filtered/included data
    print('saving html to {}'.format(os.path.join(save_root, '{}_vis'.format(split))))
    html = HTML(os.path.join(save_root, '{}_vis'.format(split)), 'subset')
    html.add_header(save_root.split('/')[-1])

    num_vis = 10
    filtered_vis = random.sample(list(filtered_data.keys()), num_vis)
    included_vis = random.sample(list(included_data.keys())[:num_per_dataset], num_vis)

    html.add_header('split: {}, threshold: {}, {} (MSE)'.format(split, str(threshold_dict), str(mse_threshold)))
    html.add_header('filtered videos, {}/{}'.format(len(filtered_data), len(filtered_data)+len(included_data)))
    ims = []
    txts = []
    links = []
    for n in filtered_vis:
        paths = filtered_data[n]['paths']
        gif_path = save_gif(paths, os.path.join(save_root, '{}_vis'.format(split), 'images'))
        ims.append(gif_path)
        txts.append(str(filtered_data[n]))
        links.append(gif_path)
    html.add_images(ims, txts, links)

    html.add_header('included videos, {}/{}'.format(len(included_data), len(filtered_data)+len(included_data)))
    ims = []
    txts = []
    links = []
    for n in included_vis:
        paths = included_data[n]['paths']
        gif_path = save_gif(paths, os.path.join(save_root, '{}_vis'.format(split), 'images'))
        ims.append(gif_path)
        txts.append(str(included_data[n]))
        links.append(gif_path)
    html.add_images(ims, txts, links)
    html.save()


if __name__ == '__main__':

    dataset_path = os.path.join(data_root, 'flowmag_raw')

    # thresholds of motion and mse
    threshold_dict = {'max_dist': 20,
                      '80th': 2,
                      '0.01st': 0.1
                      }
    mse_threshold = 10

    # path to the files of the motion and mse
    info_path_dict = {'youtube_vos': os.path.join(data_root, 'flow_data/youtube_vos_train.json'),
                      'davis': os.path.join(data_root, 'flow_data/davis_trainval.json'),
                      'vimeo': os.path.join(data_root, 'flow_data/vimeo_train.json'),
                      'tao_n1': os.path.join(data_root, 'flow_data/tao_train_n1.json'),
                      'tao_n5': os.path.join(data_root, 'flow_data/tao_train_n5.json'),
                      'uvo_n1': os.path.join(data_root, 'flow_data/uvo_train_n1.json'),
                      'uvo_n5': os.path.join(data_root, 'flow_data/uvo_train_n5.json')
                      }
    # collect train set according to the given threshold
    generate_trainset(info_path_dict, dataset_path, threshold_dict, mse_threshold)


    info_path_dict = {'youtube_vos': os.path.join(data_root, 'flow_data/youtube_vos_test.json'),
                      'davis': os.path.join(data_root, 'flow_data/davis_test.json'),
                      'vimeo': os.path.join(data_root, 'flow_data/vimeo_test.json'),
                      'tao_n1': os.path.join(data_root, 'flow_data/tao_val_n1.json'),
                      'tao_n1': os.path.join(data_root, 'flow_data/tao_val_n5.json'),
                      'uvo_n1': os.path.join(data_root, 'flow_data/uvo_test_n1.json'),
                      'uvo_n5': os.path.join(data_root, 'flow_data/uvo_test_n5.json')
                      }
    # collect test set according to the given threshold, 100 samples for every subset
    generate_testset(info_path_dict, dataset_path, threshold_dict, mse_threshold, num_per_dataset=100)

    # resize the frame pairs to get final dataset
    final_path = os.path.join(data_root, 'flowmag_data')
    resize_dataset(dataset_path, 'train', final_path)
    resize_dataset(dataset_path, 'test', final_path)
   