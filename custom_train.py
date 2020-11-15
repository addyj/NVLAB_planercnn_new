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

    epochs = options.numEpochs
    accumulate = options.accumulate  # effective bs = batch_size * accumulate = 13 * 4 = 52
    opt_img_size = options.imgSize
    opt_img_size.extend([options.imgSize[-1]] * (3 - len(options.imgSize)))
    imgsz_min, imgsz_max, imgsz_test = opt_img_size  # img sizes (min, max, test)

    # Image Sizes
    gs = 52  # (pixels) grid size
    assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
    options.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
    if options.multi_scale:
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

    yolo_config = options.cfg
    rcnn_config = PlaneConfig(options)


    ######## Plane Dataset
    # dataset = PlaneDataset(options, config, split='train', random=True)
    # dataset_test = PlaneDataset(options, config, split='test', random=False)
    #
    # print('the number of images', len(dataset))
    #
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16)

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

    ################# depth data
    # train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)


    model = POD_Model(yolo_config, rcnn_config, options)
    refine_model = RefineModel(options)

    model.cuda()
    model.train()
    refine_model.cuda()
    refine_model.train()

    # print(summary(model, input_size=(3, 416, 416)))

    refine_model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint_refine.pth'))

    start_epoch = 0
    best_fitness = 0.0

        # opt.weights = last if opt.resume else opt.weights
        # wdir = 'weights' + os.sep  # yolo weights dir
        # last = wdir + 'last.pt'
        # best = wdir + 'best.pt'
        #

    # if weights.endswith('.pt'):  # pytorch format
    #     # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
    #     chkpt = torch.load(weights, map_location=device)
    #
    #     # load model
    #     try:
    #         chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    #         model.load_state_dict(chkpt['model'], strict=False)
    #     except KeyError as e:
    #         s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
    #             "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
    #         raise KeyError(s) from e
    #
    #     # load optimizer
    #     if chkpt['optimizer'] is not None:
    #         optimizer.load_state_dict(chkpt['optimizer'])
    #         best_fitness = chkpt['best_fitness']
    #
    #     # load results
    #     if chkpt.get('training_results') is not None:
    #         with open(results_file, 'w') as file:
    #             file.write(chkpt['training_results'])  # write results.txt
    #
    #     start_epoch = chkpt['epoch'] + 1
    #     del chkpt
    #
    # elif len(weights) > 0:  # darknet format
    #     # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
    #     load_darknet_weights(model, weights)


    model_names = [name for name, param in model.named_parameters()]
    for name, param in refine_model.named_parameters():
        assert(name not in model_names)
        continue

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if v.requires_grad:
            if '.bias' in k:
                pg2 += [v]  # biases
            elif 'Conv2d.weight' in k or 'conv' in k or 'merge1.0' in k or 'merge2.0' in k:
                pg1 += [v]  # apply weight_decay
            else:
                pg0 += [v]  # all else

    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    optimizer.add_param_group({'params': refine_model.parameters()})
    del pg0, pg1, pg2

    lf = lambda x: (((1 + math.cos(
        x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)

    # Loss
    l1_criterion = nn.L1Loss()

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

    os.system('rm ' + args.test_dir + '/*.png')

    try:
        # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter()
        print("Run 'tensorboard --logdir=runs' to view tensorboard at http://localhost:6006/")
    except:
        pass

    train(args)
