"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from torch.utils.data import Dataset

import numpy as np
import time
import rcnn_utils as utils
import os
import cv2

import glob
import sys
import random as rnd
from data_prep_util import drawSegmentationImage

class PlaneDataset(Dataset):
    def __init__(self, options, config, random=True):
            # dict_keys(['XYZ', 'depth', 'mask', 'detection', 'masks', 'depth_np', 'plane_XYZ', 'depth_ori'])
        super(PlaneDataset, self).__init__()
        self.options = options
        self.config = config
        self.imagePaths = glob.glob(self.options.customDataFolder + '/images/*.png') + glob.glob(self.options.customDataFolder + '/images/*.jpg')
        if os.path.exists(self.options.customDataFolder + '/camera.txt'):
            self.camera = np.zeros(6)
            with open(self.options.customDataFolder + '/camera.txt', 'r') as f:
                for line in f:
                    values = [float(token.strip()) for token in line.split(' ') if token.strip() != '']
                    for c in range(6):
                        self.camera[c] = values[c]
                        continue
                    break
                pass
        self.camera[[0, 2, 4]] *= 640.0 / self.camera[4]
        self.camera[[1, 3, 5]] *= 480.0 / self.camera[5]
        self.anchors = utils.generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES,
                                                      self.config.RPN_ANCHOR_RATIOS,
                                                      self.config.BACKBONE_SHAPES,
                                                      self.config.BACKBONE_STRIDES,
                                                      self.config.RPN_ANCHOR_STRIDE)
        self.depthShift = 40000.0
        self.extrinsics = np.eye(4, dtype=np.float32)
        self.random = random
        self.split = 'train'

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, index):

        np.random.seed(30)

        if self.random:
            index = np.random.randint(len(self.imagePaths))
        else:
            index = index % len(self.imagePaths)
            pass

        imagePath = self.imagePaths[index]
        image = cv2.imread(imagePath)

        segmentationPath = imagePath.replace('images/', 'segmentation_masks/').replace('.png', '.npz').replace('.jpg', '.npz')
        depthPath = imagePath.replace('images/', 'depth/').replace('.jpg', '.png')
        plane_parameters = imagePath.replace('images/', 'planes/').replace('.png', '.npy').replace('.jpg', '.npy')

        try:
            depth = cv2.imread(depthPath, -1).astype(np.float32) / self.depthShift
            depth = cv2.resize(depth, (self.config.IMAGE_MAX_DIM, self.config.IMAGE_MIN_DIM))
        except:
            print('no depth image', depthPath)

        planes = np.load(plane_parameters)

        pack_segmentation = np.load(segmentationPath)
        segmentation_masks = pack_segmentation['arr_0']
        sms = np.concatenate([np.maximum(1 - segmentation_masks.sum(0, keepdims=True), 0), segmentation_masks], axis=0).transpose((1, 2, 0))
        segmentation = drawSegmentationImage(sms,blackIndex=0)

        # segmentation = np.zeros((480, 640, 3))

        # for id, plane_mask in enumerate(segmentation_masks):
        #     pmsk = cv2.cvtColor(plane_mask, cv2.COLOR_GRAY2RGB)
        #     pmsk[:,:,0] = pmsk[:,:,0] * rnd.randint(0,255) / 256
        #     pmsk[:,:,1] = pmsk[:,:,1] * rnd.randint(0,255) / 256
        #     pmsk[:,:,2] = pmsk[:,:,2] * rnd.randint(0,255) / 256
        #     segmentation += pmsk

        segmentation = (segmentation[:, :, 2] * 256 * 256 + segmentation[:, :, 1] * 256 + segmentation[:, :, 0]) // 100 - 1

        image = cv2.resize(image, (depth.shape[0], depth.shape[1]))

        # image, planes, segmentation, depth, self.camera, self.extrinsics = info
        instance_masks = []
        class_ids = []
        parameters = []

        if len(planes) > 0:
            if 'joint' in self.config.ANCHOR_TYPE:
                distances = np.linalg.norm(np.expand_dims(planes, 1) - self.config.ANCHOR_PLANES, axis=-1)
                plane_anchors = distances.argmin(-1)
            elif self.config.ANCHOR_TYPE == 'Nd':
                plane_offsets = np.linalg.norm(planes, axis=-1)
                plane_normals = planes / np.expand_dims(plane_offsets, axis=-1)
                distances_N = np.linalg.norm(np.expand_dims(plane_normals, 1) - self.config.ANCHOR_NORMALS, axis=-1)
                normal_anchors = distances_N.argmin(-1)
                distances_d = np.abs(np.expand_dims(plane_offsets, -1) - self.config.ANCHOR_OFFSETS)
                offset_anchors = distances_d.argmin(-1)
            elif 'normal' in self.config.ANCHOR_TYPE or self.config.ANCHOR_TYPE == 'patch':
                plane_offsets = np.linalg.norm(planes, axis=-1)
                plane_normals = planes / np.expand_dims(plane_offsets, axis=-1)
                distances_N = np.linalg.norm(np.expand_dims(plane_normals, 1) - self.config.ANCHOR_NORMALS, axis=-1)
                normal_anchors = distances_N.argmin(-1)
                pass
            pass

        for planeIndex, (plane, s_mask) in enumerate(zip(planes, segmentation_masks)):
            m = s_mask
            if m.sum() < 1:
                continue
            instance_masks.append(m)
            if self.config.ANCHOR_TYPE == 'none':
                class_ids.append(1)
                parameters.append(np.concatenate([plane, np.zeros(1)], axis=0))
            elif 'joint' in self.config.ANCHOR_TYPE:
                class_ids.append(plane_anchors[planeIndex] + 1)
                residual = plane - self.config.ANCHOR_PLANES[plane_anchors[planeIndex]]
                parameters.append(np.concatenate([residual, np.array([0, plane_info[planeIndex][-1]])], axis=0))
            elif self.config.ANCHOR_TYPE == 'Nd':
                class_ids.append(normal_anchors[planeIndex] * len(self.config.ANCHOR_OFFSETS) + offset_anchors[planeIndex] + 1)
                normal = plane_normals[planeIndex] - self.config.ANCHOR_NORMALS[normal_anchors[planeIndex]]
                offset = plane_offsets[planeIndex] - self.config.ANCHOR_OFFSETS[offset_anchors[planeIndex]]
                parameters.append(np.concatenate([normal, np.array([offset])], axis=0))
            elif 'normal' in self.config.ANCHOR_TYPE:
                class_ids.append(normal_anchors[planeIndex] + 1)
                normal = plane_normals[planeIndex] - self.config.ANCHOR_NORMALS[normal_anchors[planeIndex]]
                parameters.append(np.concatenate([normal, np.zeros(1)], axis=0))
            else:
                assert(False)
                pass
            continue

        parameters = np.array(parameters)

        mask = np.stack(instance_masks, axis=2)

        class_ids = np.array(class_ids, dtype=np.int32)

        image, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters = load_image_gt(self.config, index, image, depth, mask, class_ids, parameters, augment=self.split == 'train')

        ## RPN Targets
        rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors,
                                                gt_class_ids, gt_boxes, self.config)

        ## If more instances than fits in the array, sub-sample from them.
        if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]
            gt_parameters = gt_parameters[ids]
            pass

        ## Add to batch
        rpn_match = rpn_match[:, np.newaxis]
        image = utils.mold_image(image.astype(np.float32), self.config)

        depth = np.concatenate([np.zeros((80, 640)), depth, np.zeros((80, 640))], axis=0)
        segmentation = np.concatenate([np.full((80, 640), fill_value=-1, dtype=np.int32), segmentation, np.full((80, 640), fill_value=-1, dtype=np.int32)], axis=0)

        ## Convert
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image_metas = torch.from_numpy(image_metas)
        rpn_match = torch.from_numpy(rpn_match)
        rpn_bbox = torch.from_numpy(rpn_bbox).float()
        gt_class_ids = torch.from_numpy(gt_class_ids)
        gt_boxes = torch.from_numpy(gt_boxes).float()
        gt_masks = torch.from_numpy(gt_masks.astype(np.float32)).transpose(1, 2).transpose(0, 1)
        plane_indices = torch.from_numpy(gt_parameters[:, -1]).long()
        gt_parameters = torch.from_numpy(gt_parameters[:, :-1]).float()
        data_final = [image, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, depth.astype(np.float32), self.extrinsics.astype(np.float32), planes.astype(np.float32), segmentation, plane_indices]

        transformation = np.matmul(self.extrinsics, np.linalg.inv(self.extrinsics))
        rotation = transformation[:3, :3]
        translation = transformation[:3, 3]
        axis, angle = utils.rotationMatrixToAxisAngle(rotation)

        data_final.append(np.concatenate([translation, axis, np.array([angle])], axis=0).astype(np.float32))

        data_final.append(self.camera.astype(np.float32))
        return data_final

    @staticmethod
    def collate_fn(batch):
        data_fin = zip(*batch)
        return torch.stack(data_fin, 0)

def load_image_gt(config, image_id, image, depth, mask, class_ids, parameters, augment=False, use_mini_mask=True):
    """Load and return ground truth data for an image (image, mask, bounding boxes).
    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    ## Load image and mask
    shape = image.shape
    image, window, scale, padding = utils.resize_image(
        image,
        min_dim=config.IMAGE_MAX_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)

    mask = utils.resize_mask(mask, scale, padding)

    ## Random horizontal flips.
    if augment and False:
        if np.random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)
            depth = np.fliplr(depth)
            pass
        pass

    ## Bounding boxes. Note that some boxes might be all zeros
    ## if the corresponding mask got cropped out.
    ## bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)
    ## Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)
        pass

    active_class_ids = np.ones(config.NUM_CLASSES, dtype=np.int32)
    ## Image meta data
    image_meta = utils.compose_image_meta(image_id, shape, window, active_class_ids)

    if config.NUM_PARAMETER_CHANNELS > 0:
        if config.OCCLUSION:
            depth = utils.resize_mask(depth, scale, padding)
            mask_visible = utils.minimize_mask(bbox, depth, config.MINI_MASK_SHAPE)
            mask = np.stack([mask, mask_visible], axis=-1)
        else:
            depth = np.expand_dims(depth, -1)
            depth = utils.resize_mask(depth, scale, padding).squeeze(-1)
            depth = utils.minimize_depth(bbox, depth, config.MINI_MASK_SHAPE)
            mask = np.stack([mask, depth], axis=-1)
            pass
        pass
    return image, image_meta, class_ids, bbox, mask, parameters


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):

    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    ## RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    ## RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    ## Handle COCO crowds
    ## A crowd box in COCO is a bounding box around several instances. Exclude
    ## them from training. A crowd box is given a negative class ID.
    no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    ## Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    ## Match anchors to GT Boxes
    ## If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    ## If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    ## Neutral anchors are those that don't match the conditions above,
    ## and they don't influence the loss function.
    ## However, don't keep any GT box unmatched (rare, but happens). Instead,
    ## match it to the closest anchor (even if its max IoU is < 0.3).
    #
    ## 1. Set negative anchors first. They get overwritten below if a GT box is
    ## matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    ## 2. Set an anchor for each GT box (regardless of IoU value).
    ## TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    ## 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    ## Subsample to balance positive and negative anchors
    ## Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        ## Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    ## Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        ## Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    ## For positive anchors, compute shift and scale needed to transform them
    ## to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  ## index into rpn_bbox
    ## TODO: use box_refinment() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        ## Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        ## Convert coordinates to center plus width/height.
        ## GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        ## Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        ## Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        ## Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox
