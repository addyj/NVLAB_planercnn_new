import torch
from torch import optim
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

from utils import *
from visualize_utils import *
from evaluate_utils import *
from options import parse_args
from config import PlaneConfig

from datasets.inference_dataset import InferenceDataset
from plane_utils import *
from options import parse_args

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def train(options):
    # dict_keys(['XYZ', 'depth', 'mask', 'detection', 'masks', 'depth_np', 'plane_XYZ', 'depth_ori'])
    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    config = PlaneConfig(options)
    config.FITTING_TYPE = options.numAnchorPlanes

    if 'custom' in options.dataset:
        image_list = glob.glob(options.customDataFolder + '/*.png') + glob.glob(options.customDataFolder + '/*.jpg')
        if os.path.exists(options.customDataFolder + '/camera.txt'):
            camera = np.zeros(6)
            with open(options.customDataFolder + '/camera.txt', 'r') as f:
                for line in f:
                    values = [float(token.strip()) for token in line.split(' ') if token.strip() != '']
                    for c in range(6):
                        camera[c] = values[c]
                        continue
                    break
                pass
        else:
            camera = [filename.replace('.png', '.txt').replace('.jpg', '.txt') for filename in image_list]
            pass
        dataset = InferenceDataset(options, config, image_list=image_list, camera=camera, random=True)
        pass

    print('the number of images', len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16)

    model = POD_Model(config, options)
    refine_model = RefineModel(options)

    model.cuda()
    refine_model.cuda()
    model.train()
    refine_model.train()
    
    print(summary(model, input_size=(3, 384, 384)))

    refine_model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint_refine.pth'))

    model_names = [name for name, param in model.named_parameters()]
    for name, param in refine_model.named_parameters():
        assert(name not in model_names)
        continue

    # trainables_wo_bn = [param for name, param in model.named_parameters() if param.requires_grad and not 'bn' in name]
    # trainables_only_bn = [param for name, param in model.named_parameters() if param.requires_grad and 'bn' in name]
    #
    # optimizer = optim.SGD([
    #     {'params': trainables_wo_bn, 'weight_decay': 0.0001},
    #     {'params': trainables_only_bn},
    #     {'params': refine_model.parameters()}
    # ], lr=options.LR, momentum=0.9)


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
    print('keyname=%s task=%s started'%(args.keyname, args.task))

    train(args)
