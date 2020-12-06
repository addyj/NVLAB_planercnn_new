import torch
import torch.nn as nn
from options import parse_args
import datetime
import os
import numpy as np
from .midas import Encoder, Decoder1
from .rcnn import Decoder3
from .yolo_models import Decoder2

class POD_Model(nn.Module):
    def __init__(self, yolo_config, rcnn_config, options, model_dir='test'):
        super(POD_Model, self).__init__()
        self.rcnn_config = rcnn_config
        self.yolo_config = yolo_config
        self.model_dir = model_dir
        self.loss_history = []
        self.val_loss_history = []
        # self.set_log_dir()

        self.options = options
        self.encoder = Encoder()

        self.decoder1 = Decoder1()

        self.m_y_merge1 = nn.Sequential(nn.Conv2d(in_channels=1024,
                                            out_channels=512,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            bias=False),
                                    nn.BatchNorm2d(512, momentum=0.03, eps=1E-4),
                                    nn.LeakyReLU(0.1, inplace=True)
                                )
        self.m_y_merge2 = nn.Sequential(nn.Conv2d(in_channels=512,
                                            out_channels=256,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            bias=False),
                                    nn.BatchNorm2d(256, momentum=0.03, eps=1E-4),
                                    nn.LeakyReLU(0.1, inplace=True)
                                )
        self.m_y_merge3 = nn.Sequential(nn.Conv2d(in_channels=2048,
                                            out_channels=1024,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            bias=False),
                                    nn.BatchNorm2d(1024, momentum=0.03, eps=1E-4),
                                    nn.LeakyReLU(0.1, inplace=True)
                                )

        self.decoder2 = Decoder2(self.yolo_config)

        self.decoder3 = Decoder3(self.rcnn_config)
        
    def forward(self, x, plane_data):
        c1, c2, c3, c4 = self.encoder(x)
        #384 torch.Size([2, 256, 96, 96]) torch.Size([2, 512, 48, 48]) torch.Size([2, 1024, 24, 24]) torch.Size([2, 2048, 12, 12])
        #(skip and concat n send to yolo)
        #416 torch.Size([2, 256, 104, 104]) torch.Size([2, 512, 52, 52]) torch.Size([2, 1024, 26, 26]) torch.Size([2, 2048, 13, 13])
        #640 size for planercnn compatibility
        midas_output = self.decoder1(c1, c2, c3, c4)
        m_y_c3 = self.m_y_merge1(c3)
        m_y_c2 = self.m_y_merge2(c2)
        m_y_c4 = self.m_y_merge3(c4)
        yolo_output = self.decoder2(m_y_c4, m_y_c2, m_y_c3)

        plane_output = []
        for pl_pred_idx in range(self.options.batchSize):

            camera = torch.from_numpy(plane_data[pl_pred_idx][14]).cuda()

            encoder_ext = [c1[pl_pred_idx].unsqueeze(0), c2[pl_pred_idx].unsqueeze(0), c3[pl_pred_idx].unsqueeze(0), c4[pl_pred_idx].unsqueeze(0)]
            image_metas = plane_data[pl_pred_idx][1].numpy()
            rpn_match = plane_data[pl_pred_idx][2].cuda()
            rpn_bbox = plane_data[pl_pred_idx][3].cuda()
            gt_class_ids = plane_data[pl_pred_idx][4].cuda()
            gt_boxes = plane_data[pl_pred_idx][5].cuda()
            gt_masks = plane_data[pl_pred_idx][6].cuda()
            gt_parameters = plane_data[pl_pred_idx][7].cuda()
            gt_depth = torch.from_numpy(plane_data[pl_pred_idx][8]).cuda()
            extrinsics = torch.from_numpy(plane_data[pl_pred_idx][9]).cuda()
            gt_plane = torch.from_numpy(plane_data[pl_pred_idx][10]).cuda()
            gt_segmentation = torch.from_numpy(plane_data[pl_pred_idx][11]).cuda()
            plane_indices = plane_data[pl_pred_idx][12].cuda()

            plane_predict = self.decoder3.predict([encoder_ext, np.expand_dims(image_metas, axis=0), gt_class_ids.unsqueeze(0), gt_boxes.unsqueeze(0), gt_masks.unsqueeze(0), gt_parameters.unsqueeze(0), camera.unsqueeze(0)], mode='training_detection', use_nms=2, use_refinement=False, return_feature_map=True)
            plane_output.append(plane_predict)

        return midas_output, yolo_output, plane_output

    # def set_log_dir(self, model_path=None):
    #     """Sets the model log directory and epoch counter.
    #
    #     model_path: If None, or a format different from what this code uses
    #     then set a new log directory and start epochs from 0. Otherwise,
    #     extract the log directory and the epoch counter from the file
    #     name.
    #     """
    #
    #     ## Set date and epoch counter as if starting a new model
    #     self.epoch = 0
    #     now = datetime.datetime.now()
    #
    #     ## If we have a model path with date and epochs use them
    #     if model_path:
    #         ## Continue from we left of. Get epoch and date from the file name
    #         ## A sample model path might look like:
    #         ## /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
    #         regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.pth"
    #         m = re.match(regex, model_path)
    #         if m:
    #             now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
    #             int(m.group(4)), int(m.group(5)))
    #             self.epoch = int(m.group(6))
    #
    #             ## Directory for training logs
    #             self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
    #             self.rcnn_config.NAME.lower(), now))
    #
    #             ## Path to save after each epoch. Include placeholders that get filled by Keras.
    #             self.checkpoint_path = os.path.join(self.log_dir, "pod_pred_{}_*epoch*.pth".format(
    #             self.rcnn_config.NAME.lower()))
    #             self.checkpoint_path = self.checkpoint_path.replace(
    #         "*epoch*", "{:04d}")
