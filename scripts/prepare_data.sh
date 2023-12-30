ROOT=./data

mkdir $ROOT/davis2017
mkdir $ROOT/davis2017/DAVIS-2017-trainval-480p
mkdir $ROOT/davis2017/DAVIS-2017-test-dev-480p
wget -P $ROOT/davis2017 https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
wget -P $ROOT/davis2017 https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip
unzip -q $ROOT/davis2017/DAVIS-2017-trainval-480p.zip -d $ROOT/davis2017/DAVIS-2017-trainval-480p
unzip -q $ROOT/davis2017/DAVIS-2017-test-dev-480p.zip -d $ROOT/davis2017/DAVIS-2017-test-dev-480p
rm $ROOT/davis2017/DAVIS-2017-trainval-480p.zip
rm $ROOT/davis2017/DAVIS-2017-test-dev-480p.zip
python scripts/generate_davis_subset.py --data_root $ROOT

mkdir $ROOT/tao
wget -P $ROOT/tao https://motchallenge.net/data/1-TAO_TRAIN.zip
wget -P $ROOT/tao https://motchallenge.net/data/2-TAO_VAL.zip
unzip -q $ROOT/tao/1-TAO_TRAIN.zip -d $ROOT/tao
unzip -q $ROOT/tao/2-TAO_VAL.zip -d $ROOT/tao
rm $ROOT/tao/1-TAO_TRAIN.zip
rm $ROOT/tao/2-TAO_VAL.zip
python scripts/generate_tao_subset.py --data_root $ROOT


mkdir $ROOT/uvo
mkdir $ROOT/uvo_frames
gdown 1JKbn3cHk1NY_v4s1-jdOz4c0VflKI892 -O $ROOT/uvo_frames/UVO_video_train_dense.json
gdown 1D2B6Qk1WEZAkqJPPQonxa5WiA0bd2DNA -O $ROOT/uvo_frames/UVO_video_test_dense.json
gdown 1wQHv0IF3oXe4dawPiLoPxWD9JCAa3pro -O $ROOT/uvo/uvo_videos_dense.zip
unzip -q $ROOT/uvo/uvo_videos_dense.zip -d $ROOT/uvo
rm $ROOT/uvo/uvo_videos_dense.zip
python scripts/save_uvo_frames.py --data_root $ROOT
python scripts/generate_uvo_subset.py --data_root $ROOT


mkdir $ROOT/vimeo
wget -P $ROOT/vimeo http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip
unzip -q $ROOT/vimeo/vimeo_septuplet.zip -d $ROOT/vimeo
rm $ROOT/vimeo/vimeo_septuplet.zip
python scripts/generate_vimeo_subset.py --data_root $ROOT


mkdir $ROOT/youtube-vos-2019
gdown 13Eqw0gVK-AO5B-cqvJ203mZ2vzWck9s4 -O $ROOT/youtube-vos-2019/train.zip
gdown 1S50D-vwOKrmTJNh6VDfXhj8L0jkrNA6V -O $ROOT/youtube-vos-2019/test.zip
unzip -q $ROOT/youtube-vos-2019/train.zip -d $ROOT/youtube-vos-2019
unzip -q $ROOT/youtube-vos-2019/test.zip -d $ROOT/youtube-vos-2019
rm $ROOT/youtube-vos-2019/train.zip
rm $ROOT/youtube-vos-2019/test.zip
python scripts/generate_yvos_subset.py --data_root $ROOT

python scripts/collect_subsets.py --data_root $ROOT
