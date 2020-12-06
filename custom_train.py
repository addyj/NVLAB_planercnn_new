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
# from models.refinement_net import *
from models.modules import *

from rcnn_utils import *
from visualize_utils import *
from evaluate_utils import *
from config import PlaneConfig

from plane_utils import *
from midas_utils import ssim
from options import parse_args
import torch.nn.functional as F

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

    batch_size = options.batchSize
    epochs = options.numEpochs
    accumulate = options.accumulate  # effective bs = batch_size * accumulate = 13 * 4 = 52
    opt_img_size = options.imgSize
    opt_img_size.extend([options.imgSize[-1]] * (3 - len(options.imgSize)))
    imgsz_min, imgsz_max, imgsz_test = opt_img_size  # img sizes (min, max, test)

    # Image Sizes
    # gs = 52  # (pixels) grid size
    # assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
    # options.multiScale |= imgsz_min != imgsz_max  # multi if different (min, max)
    # if options.multiScale:
    #     if imgsz_min == imgsz_max:
    #         imgsz_min //= 1.5
    #         imgsz_max //= 0.667
    #     grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
    #     imgsz_min, imgsz_max = grid_min * gs, grid_max * gs
    img_size = imgsz_max  # initialize with max size

    init_seeds(seed=30)

    # Remove previous results
    results_file = 'yolo_results.txt'
    for f in glob.glob('*_batch*.png') + glob.glob(results_file):
        os.remove(f)

    yolo_config = options.cfg
    rcnn_config = PlaneConfig(options)

    data = options.data
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = int(data_dict['classes'])  # number of classes
    hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

    # Dataset
    dataset = LoadImagesAndLabels(options, rcnn_config, train_path, img_size, batch_size,
                                  augment=False,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=options.rect  # rectangular training
                                  )

    # # Dataloader
    nw = 4 # number of workers
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=nw,
                            shuffle=not options.rect,  # Shuffle=True unless rectangular training is used
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    # # Testloader
    # testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
    #                                                              hyp=hyp,
    #                                                              rect=True,
    #                                                              cache_images=opt.cache_images,
    #                                                              single_cls=opt.single_cls),
    #                                          batch_size=batch_size,
    #                                          num_workers=nw,
    #                                          pin_memory=True,
    #                                          collate_fn=dataset.collate_fn)
    #

    model = POD_Model(yolo_config, rcnn_config, options)
    # refine_model = RefineModel(options)

    model.cuda()
    model.train()
    # refine_model.cuda()
    # refine_model.train()

    # refine_model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint_refine.pth'))

    start_epoch = 0
    best_fitness = 0.0

    # opt.weights = last if opt.resume else opt.weights
    wdir = 'weights' + os.sep  # yolo weights dir
    # last = wdir + 'last.pt'
    # best = wdir + 'best.pt'

    midas_state_dict = torch.hub.load_state_dict_from_url(
          "https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.pt", progress=True, check_hash=True
      )

    model.encoder.load_state_dict(midas_state_dict, strict=False)
    model.decoder1.load_state_dict(midas_state_dict, strict=False)

    chkpt = torch.load('weights/last2.pt')
    yolo_extract = dict()
    for k, v in chkpt['model'].items():
      module_key = k.split('.')
      if int(module_key[1]) > 74:
        module_key[1] = str(int(module_key[1]) - 75)
        yolo_extract['.'.join(module_key)] = v

    model.decoder2.load_state_dict(yolo_extract, strict=False)

    rcnn_state_dict = torch.load(options.checkpoint_dir + '/checkpoint.pth')
    for key in list(rcnn_state_dict.keys()):
        if key.startswith('fpn.C'):
           del rcnn_state_dict[key]

    model.decoder3.load_state_dict(rcnn_state_dict, strict = False)
    model.decoder3.set_trainable(r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)")

    if chkpt['optimizer'] is not None:
        # optimizer.load_state_dict(chkpt['optimizer'])
        best_fitness = chkpt['best_fitness']

    # # load results
    # if chkpt.get('training_results') is not None:
    #     with open(results_file, 'w') as file:
    #         file.write(chkpt['training_results'])  # write results.txt

    del chkpt
    del yolo_extract
    del midas_state_dict
    del rcnn_state_dict


    # model_names = [name for name, param in model.named_parameters()]
    # for name, param in refine_model.named_parameters():
    #     assert(name not in model_names)
    #     continue

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if v.requires_grad:
            if '.bias' in k:
                pg2 += [v]  # biases
            elif 'Conv2d.weight' in k or 'conv' in k or 'merge1.0' in k or 'merge2.0' in k or 'merge3.0' in k:
                pg1 += [v]  # apply weight_decay
            else:
                pg0 += [v]  # all else

    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    # optimizer.add_param_group({'params': refine_model.parameters()})
    del pg0, pg1, pg2

    lf = lambda x: (((1 + math.cos(
        x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)

    # Loss
    l1_criterion = nn.L1Loss()

    # Model parameters for YOLO
    model.decoder2.nc = nc  # attach number of classes to model
    model.decoder2.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.decoder2.gr = model.gr
    model.class_weights = labels_to_class_weights(dataset.labels, nc).cuda()  # attach class weights

    # Model EMA
    ema = torch_utils.ModelEMA(model)

    # Start training
    nb = len(dataloader)  # number of batches
    print("Numbers of Batches: ", nb)
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)

    for epoch in range(start_epoch, epochs):
        model.train()

        mloss = torch.zeros(4).cuda()  # mean losses
        print(('\n' + '%12s' * 10) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'yolo_loss', 'targets', 'img_size', 'depth_loss', 'plane_loss'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, shapes, planedata) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.cuda().float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.cuda()

            # Burn-in
            if ni <= n_burn * 2:
                model.gr = np.interp(ni, [0, n_burn * 2], [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                model.decoder2.gr = model.gr
                if ni == n_burn:  # burnin complete
                    print_model_biases(model)

                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, [0, n_burn], [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, [0, n_burn], [0.9, hyp['momentum']])

            # # Multi-Scale training
            # if opt.multi_scale:
            #     if ni / accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
            #         img_size = random.randrange(grid_min, grid_max + 1) * gs
            #     sf = img_size / max(imgs.shape[2:])  # scale factor
            #     if sf != 1:
            #         ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
            #         imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Run model
            midas_out, yolo_out, plane_out = model(imgs, planedata)

            plane_losses = []
            depth_losses = []
            for batch_idx in range(batch_size):

                rpn_match = planedata[batch_idx][2].cuda()
                rpn_bbox = planedata[batch_idx][3].cuda()
                gt_depth = torch.from_numpy(planedata[batch_idx][8]).cuda()

                rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, detections, detection_masks, detection_gt_parameters, detection_gt_masks, rpn_rois, roi_features, roi_indices, feature_map, depth_np_pred = plane_out[batch_idx]

                ## Plane losses
                rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, mrcnn_parameter_loss = compute_losses(rcnn_config, rpn_match.unsqueeze(0), rpn_bbox.unsqueeze(0), rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters)

                plane_losses += [rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + mrcnn_parameter_loss]

                ### Midas losses
                l_depth = l1_criterion(midas_out[batch_idx], gt_depth)
                # l_ssim = torch.clamp((1 - ssim(midas_out[batch_idx].unsqueeze(0).unsqueeze(0), gt_depth.unsqueeze(0).unsqueeze(0), val_range = 1000.0 / 10.0)) * 0.5, 0, 1)
                l_mse = F.mse_loss(midas_out[batch_idx], gt_depth)
                d_loss = (1.0 * l_mse) + (0.1 * l_depth)
                depth_losses +=[d_loss]

                gt_depth = gt_depth.unsqueeze(0)
                depth_np_loss = l1LossMask(depth_np_pred[:, 80:560], gt_depth[:, 80:560], (gt_depth[:, 80:560] > 1e-4).float())
                plane_losses.append(depth_np_loss)
                normal_np_pred = None

            plane_batch_loss = sum(plane_losses)
            depth_batch_loss = sum(depth_losses)
            ### Yolo loss
            yolo_loss, loss_items = compute_loss(yolo_out, targets, model.decoder2)
            # if not torch.isfinite(yolo_loss):
            #     print('WARNING: non-finite loss, ending training ', loss_items)
            #     return results

            # Scale loss by nominal batch_size of 64
            yolo_loss *= batch_size / 64
            total_loss = depth_batch_loss + yolo_loss + plane_batch_loss
            # Compute gradient
            total_loss.backward()

            # Optimize accumulated gradient
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%12s' * 2 + '%12.3g' * 8) % ('%g/%g' % (epoch+1, epochs), mem, *mloss, len(targets), img_size, depth_batch_loss, plane_batch_loss)
            pbar.set_description(s)

            # Plot images with bounding boxes
            # if ni < 1:
            #     f = 'train_batch%g.png' % i  # filename
            #     plot_images(imgs=imgs, targets=targets, paths=paths, fname=f)
                # if tb_writer:
                #     tb_writer.add_image(f, cv2.imread(f)[:, :, ::-1], dataformats='HWC')
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

        # # Process epoch results
        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs
        # if not opt.notest or final_epoch:  # Calculate mAP
        #     is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
        #     results, maps = test.test(cfg,
        #                               data,
        #                               batch_size=batch_size,
        #                               img_size=imgsz_test,
        #                               model=ema.ema,
        #                               save_json=final_epoch and is_coco,
        #                               single_cls=opt.single_cls,
        #                               dataloader=testloader)
        #
        # # Write epoch results
        # with open(results_file, 'a') as f:
        #     f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        # # Update best mAP
        # fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        # if fi > best_fitness:
        #     best_fitness = fi

        # Save training results
        save = (not options.nosave) or (final_epoch)
        if save:
            # with open(results_file, 'r') as f:
            #     # Create checkpoint
            #     _chkpt = {'epoch': epoch,
            #              'best_fitness': best_fitness,
            #              'training_results': f.read(),
            #              'model': ema.ema.module.state_dict() if hasattr(model, 'module') else ema.ema.state_dict(),
            #              'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last checkpoint
            torch.save(model.state_dict(), wdir + 'last_wt.pt')

            # Save best checkpoint
            # if (best_fitness == fi) and not final_epoch:
            #     torch.save(_chkpt, best)

            # Save backup every 10 epochs (optional)
            # if epoch > 0 and epoch % 10 == 0:
            #     torch.save(_chkpt, wdir + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del _chkpt

        # end epoch ----------------------------------------------------------------------------------------------------

    plot_results()
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    torch.cuda.empty_cache()

if __name__ == '__main__':
    args = parse_args()

    args.keyname = 'trident'

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

    # try:
    #     # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
    #     from torch.utils.tensorboard import SummaryWriter
    #     tb_writer = SummaryWriter()
    #     print("Run 'tensorboard --logdir=runs' to view tensorboard at http://localhost:6006/")
    # except:
    #     pass

    train(args)
