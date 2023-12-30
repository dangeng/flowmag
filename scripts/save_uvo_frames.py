import os
import cv2
import tqdm
import glob
import argparse

if __name__ == '__main__':
    # save videos in uvo_videos_dense as image files
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--data_root', type=str, help='root folder to save data')
    args = parser.parse_args()
    data_root = args.data_root

    path = os.path.join(data_root, 'uvo')
    save_root = os.path.join(data_root, 'uvo_frames')
    split = 'uvo_videos_dense'
    
    all_videos = glob.glob(os.path.join(path, split, '*.mp4'))
    for video in tqdm.tqdm(all_videos):
        fn = os.path.basename(video).split('.mp4')[0]
        os.makedirs(os.path.join(save_root, split, fn))
        cap = cv2.VideoCapture(video)
        ret, frame = cap.read()
        i = 0
        while ret:
            cv2.imwrite(os.path.join(save_root, split, fn, '{}.png'.format(str(i).zfill(5))), frame)
            i += 1
            ret, frame = cap.read()
        cap.release()
