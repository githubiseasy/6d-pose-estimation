@echo on
set ID="beaker1"
python docker_images/src/prepare.py %ID%
python docker_images/src/check.py --datacfg data/%ID%/%ID%.data
: python docker_images/src/train.py --datacfg data/%ID%/%ID%.data --modelcfg data/%ID%/models/yolo-pose.cfg --initweightfile datasets/models/darknet19_448.conv.23 --pretrain_num_epochs 15
:: python docker_images/src/train.py --datacfg data/%ID%/%ID%.data --modelcfg data/%ID%/models/yolo-pose.cfg --initweightfile data/%ID%/models/model.weights --pretrain_num_epochs 15
:: python docker_images/src/valid.py --datacfg data/%ID%/%ID%.data --modelcfg data/%ID%/models/yolo-pose.cfg --weightfile data/%ID%/models/model.weights
