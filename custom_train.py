import torch
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import summary

import os
from tqdm import tqdm
import numpy as np
import cv2
import sys
import glob

from models.rcnn import *
from models.custom_model import *
from models.refinement_net import *
from models.modules import *
from datasets.plane_stereo_dataset import *

from rcnn_utils import *
from visualize_utils import *
from evaluate_utils import *
from config import PlaneConfig

from datasets.inference_dataset import InferenceDataset
from plane_utils import *
from options import parse_args

from models.yolo_models import *
from yolo_utils.datasets import *
from yolo_utils.utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# YOLO_V3 Hyperparameters
hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

def train(options):
    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    accumulate = options.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    opt_img_size = options.img_size.extend([options.img_size[-1]] * (3 - len(options.img_size)))
    imgsz_min, imgsz_max, imgsz_test = opt_img_size  # img sizes (min, max, test)

    # Image Sizes
    gs = 52  # (pixels) grid size
    assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
    opt.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
    if opt.multi_scale:
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.5
            imgsz_max //= 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = grid_min * gs, grid_max * gs
    img_size = imgsz_max  # initialize with max size

    init_seeds(seed=30)

    # Remove previous results
    results_file = 'yolo_results.txt'
    for f in glob.glob('*_batch*.png') + glob.glob(results_file):
        os.remove(f)



    # opt.weights = last if opt.resume else opt.weights
    # wdir = 'weights' + os.sep  # yolo weights dir
    # last = wdir + 'last.pt'
    # best = wdir + 'best.pt'
    #

    yolo_config = options.cfg
    rcnn_config = PlaneConfig(options)


    ######## Plane Dataset
    # dict_keys(['XYZ', 'depth', 'mask', 'detection', 'masks', 'depth_np', 'plane_XYZ', 'depth_ori'])

    ### InferenceDataset for Plane doesnt work correctly
    # rcnn_config.FITTING_TYPE = options.numAnchorPlanes
    #
    # if 'custom' in options.dataset:
    #     image_list = glob.glob(options.customDataFolder + '/*.png') + glob.glob(options.customDataFolder + '/*.jpg')
    #     if os.path.exists(options.customDataFolder + '/camera.txt'):
    #         camera = np.zeros(6)
    #         with open(options.customDataFolder + '/camera.txt', 'r') as f:
    #             for line in f:
    #                 values = [float(token.strip()) for token in line.split(' ') if token.strip() != '']
    #                 for c in range(6):
    #                     camera[c] = values[c]
    #                     continue
    #                 break
    #             pass
    #     else:
    #         camera = [filename.replace('.png', '.txt').replace('.jpg', '.txt') for filename in image_list]
    #         pass
    #     dataset = InferenceDataset(options, rcnn_config, image_list=image_list, camera=camera, random=True)
    #     pass
    #
    # print('the number of images', len(dataset))
    #
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16)

    ################ yolo data

    # data = opt.data
    # data_dict = parse_data_cfg(data)
    # train_path = data_dict['train']
    # test_path = data_dict['valid']
    # nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes
    # hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset


    model = POD_Model(yolo_config, rcnn_config, options)
    refine_model = RefineModel(options)

    model.cuda()
    model.train()
    refine_model.cuda()
    refine_model.train()

    # print(summary(model, input_size=(3, 416, 416)))

    refine_model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint_refine.pth'))


    # weights = options.weights  # initial training weights

    model_names = [name for name, param in model.named_parameters()]
    for name, param in refine_model.named_parameters():
        assert(name not in model_names)
        continue

    trainables_wo_bn = [name for name, param in model.named_parameters() if param.requires_grad and not 'bn' in name]
    print(trainables_wo_bn)
    # trainables_only_bn = [param for name, param in model.named_parameters() if param.requires_grad and 'bn' in name]
    #
    # optimizer = optim.SGD([
    #     {'params': trainables_wo_bn, 'weight_decay': 0.0001},
    #     {'params': trainables_only_bn},
    #     {'params': refine_model.parameters()}
    # ], lr=options.LR, momentum=0.9)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [k]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [k]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    print(">>>>>>   ", pg2)
    print(">>>>>>   ", pg1)
    # optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    # optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    # optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    # optimizer.add_param_group({'params': refine_model.parameters()})
    # del pg0, pg1, pg2



if __name__ == '__main__':
    args = parse_args()

    args.keyname = 'planercnn'

    args.keyname += '_' + args.anchorType
    if args.dataset != '':
        args.keyname += '_' + args.dataset
        pass
    if args.trainingMode != 'all':
        args.keyname += '_' + args.trainingMode
        pass
    if args.suffix != '':
        args.keyname += '_' + args.suffix
        pass

    args.checkpoint_dir = 'checkpoint/' + args.keyname
    args.test_dir = 'test/' + args.keyname

    if False:
        writeHTML(args.test_dir, ['image_0', 'segmentation_0', 'depth_0', 'depth_0_detection', 'depth_0_detection_ori'], labels=['input', 'segmentation', 'gt', 'before', 'after'], numImages=20, image_width=160, convertToImage=True)
        exit(1)

    os.system('rm ' + args.test_dir + '/*.png')

    try:
        # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter()
        print("Run 'tensorboard --logdir=runs' to view tensorboard at http://localhost:6006/")
    except:
        pass

    train(args)
