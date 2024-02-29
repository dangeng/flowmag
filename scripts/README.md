# Flowmag Dataset Collection

Set the root directory to store data in `scripts/prepare_data.sh` and run `sh scripts/prepare_data.sh` to collect the dataset named `flowmag_data` under the root directory. The default directory is `./data`. Please ensure that your root directory has sufficient storage space.

We provide the file containing the information of our filtered frames for train set, named [train_included.json](https://drive.google.com/file/d/1-3zWkw6IOmc9JysL-G7und0cM-sDXt7P/view?usp=sharing).
If you prefer reproducing the train set with our threshold, you might download the five datasets, process UVO (save as frames), and
collect frame pairs with this file and `collect_trainset_from_json()` in `collect_subsets.py`.

We provide the zip file of [test data](https://drive.google.com/file/d/1e9KljPpIHB5Yq8r2-XcHLEHlym6n1H5C/view?usp=sharing), 
containing a folder of images named `test` and a json file of image filenames named `test_fn.json`.