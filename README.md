# Trident Detection: Plane, Object and Depth
## Introduction

A deep neural architecture, which combines PlaneR-CNN, YOLOv3 and Midas architecture to detects arbitrary number of planes, depth and bounding boxes for a image.

For train log [Colab Train](https://colab.research.google.com/drive/1dzrYak3wLTHlpLf_I1viiH7EtTpwSj2E?usp=sharing)
Repositories referred and merged [PlaneRCNN](https://github.com/NVlabs/planercnn), [YoloV3](https://github.com/theschoolofai/YoloV3) and [MiDas](https://github.com/intel-isl/MiDaS)
The code is implemented using PyTorch.

## Getting Started 
Clone repository: 
```
git clone https://github.com/Trident_Detection.git
```

Build RoI Align module
```
cd Trident_Detection/roialign/
python setup.py install
cd ..
```

For **nms** we use torchvision inbuilt ops and for **roialign** refer [RoIAlign Repo](https://github.com/longcw/RoIAlign.pytorch)

Make **checkpoint/** folder for plane-rcnn and **weights/** folder for yolov3 pretrained weigths. For midas the pretrained weights are directly loaded url.

Once you have a saved pretrained weight file for this model you can comment out the load code and load the pretrained wts file. [automation for this is not available for now.]

Load custom data into **data/** folder. Refer YOLOv3 dataset structure construction. Refer the custom data set for reference [Trident Data Full](https://drive.google.com/file/d/1Oia4b6dNvxs9TbFCs8_60VS2iWxcHgeW/view?usp=sharing) and [Trident Data](https://drive.google.com/file/d/1orpEeyaq9LExJ_MyH7XCAQyxTcTxzTpJ/view?usp=sharing) . Also follow the extension according to the custom dataset.

## Model

### MiDAS

Took  MiDAS ResNext101 as the backbone network for feature extractor acting as Encoder, and the stages of  ResNext101 are skipped to Midas Decoder, YoloV3 and PlaneRCNN.
MiDAS both encoder and Decoder layers were frozen.

### YoloV3

For YoloV3 model integration, the original configuration file was bisected from layer 75 and added to Midas encoder for bounding box prediction. [New YOLO v3 Config File](https://github.com/addyj/Trident_Detection/blob/master/cfg/yolov3-custom.cfg)

As MiDAS Encoder's - ResNext101 and  YoloV3's - DarkNet are different, added additional layers were added between Encoder and Yolo predictor to match the dimensions.

### PlaneRCNN

Same as Yolo the Midas Encoder outs from ResNext101 were feeded to PlaneRCNN Feature Pyramid Network (FPN) and no intermidiate layers were added as the dimension matched to the PlaneRCNN Resnet101. 
Note: Should add some intermidiate layers for proper merging.


## Training
### Training data preparation
Please first download the ScanNet dataset (v2), unzip it to "$ROOT_FOLDER/scans/", and extract image frames from the *.sens* file using the official [reader](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py).

Then download our plane annotation from [here](https://www.dropbox.com/s/u2wl4ji700u4shq/ScanNet_planes.zip?dl=0), and merge the "scans/" folder with "$ROOT_FOLDER/scans/". (If you prefer other locations, please change the paths in *datasets/scannet_scene.py*.)

After the above steps, ground truth plane annotations are stored under "$ROOT_FOLDER/scans/scene*/annotation/". Among the annotations, *planes.npy* stores the plane parameters which are represented in the global frame. Plane segmentation for each image view is stored under *segmentation/*.

To generate such training data on your own, please refer to *data_prep/parse.py*. Please refer to the README under *data_prep/* for compilation.

Besides scene-specific annotation under each scene folder, please download global metadata from [here](https://www.dropbox.com/s/v7qb7hwas1j766r/metadata.zip?dl=0), and unzip it to "$ROOT_FOLDER". Metadata includes the normal anchors (anchor_planes_N.npy) and invalid image indices caused by tracking issues (invalid_indices_*.txt). 

### Training with custom data
To train on custom data, you need a list of planes, where each plane is represented using three parameters (as explained above) and a 2D binary mask. In our implementation, we use one 2D segmentation map where pixels with value *i* belong to the *i*th plane in the list. The easiest way is to replace the ScanNetScene class with something interacts with your custom data. Note that, the plane_info, which stores some semantic information and global plane index in the scene, is not used in this project. The code is misleading as global plane indices are read from plane_info [here](https://github.com/NVlabs/planercnn/blob/01e03fe5a97b7afc4c5c4c3090ddc9da41c071bd/datasets/plane_stereo_dataset.py#L194), but they are used only for debugging purposes.

### Training script
```bash
python train_planercnn.py --restore=2 --suffix=warping_refine
```
options:
```bash
--restore:
- 0: training from scratch (not tested)
- 1 (default): resume training from saved checkpoint
- 2: training from pre-trained mask-rcnn model

--suffix (the below arguments can be concatenated):
- '': training the basic version
- 'warping': with the warping loss
- 'refine': with the refinement network
- 'refine_only': train only the refinement work
- 'warping_refine_after': add the warping loss after the refinement network instead of appending both independently

--anchorType:
- 'normal' (default): regress normal using 7 anchors
- 'normal[k]' (e.g., normal5): regress normal using k anchors, normal0 will regress normal directly without anchors
- 'joint': regress final plane parameters directly instead of predicting normals and depthmap separately
```

Temporary results are written under *test/* for debugging purposes.

## Evaluation
To evaluate the performance against existing methods, please run:
```bash
python evaluate.py --methods=f --suffix=warping_refine
```
Options:
```bash
--methods:
- f: evaluate PlaneRCNN (use --suffix and --anchorType to specify configuration as explained above)
- p: evaluate PlaneNet
- e: evaluate PlaneRecover
- t: evaluate MWS (--suffix=gt for MWS-G)
```
Statistics are printed in terminal and saved in *logs/global.txt* for later analysis.

Note that [PlaneNet](https://github.com/art-programmer/PlaneNet/blob/master/LICENSE) and [PlaneRecover](https://github.com/fuy34/planerecover/blob/master/LICENSE) are under the MIT license.

To evaluate on the NYU Depth dataset, please first download the labeled dataset from the official [website](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), and the official train/test split from [here](http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat). Put them under the same folder "$NYU_FOLDER". To evaluate, please run,
```bash
python evaluate.py --methods=f --suffix=warping_refine --dataset=nyu --dataFolder="$NYU_FOLDER"
```

Note that the numbers are off with the provided model. We retrained the model after cleaning up the code, which is different from the model we tested for the publication.



