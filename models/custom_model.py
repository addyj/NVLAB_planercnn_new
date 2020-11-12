import torch
import torch.nn as nn
from options import parse_args
import datetime
from .midas import Encoder, Decoder1
from .rcnn import Decoder3

class POD_Model(nn.Module):
    def __init__(self, config, options, model_dir='test'):
        super(POD_Model, self).__init__()
        self.model_dir = model_dir
        self.loss_history = []
        self.val_loss_history = []
        self.set_log_dir()

        self.midas_checkpoint = "https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.pt"
        self.midas_state_dict = torch.hub.load_state_dict_from_url(
              self.midas_checkpoint, progress=True, check_hash=True
          )
        self.config = config
        self.options = options
        self.encoder = Encoder(self.midas_state_dict)
        self.decoder1 = Decoder1(self.midas_state_dict)
        # self.decoder2 = Decoder2()
        self.decoder3 = Decoder3(self.config,
                                self.encoder.pretrained.layer1,
                                self.encoder.pretrained.layer2,
                                self.encoder.pretrained.layer3,
                                self.encoder.pretrained.layer4
                                )
        print(torch.load(options.checkpoint_dir + '/checkpoint.pth'))
        # self.decoder3.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint.pth'), strict = False)
        self.decoder3.set_trainable(r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)")
        
    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

        ## Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        ## If we have a model path with date and epochs use them
        if model_path:
            ## Continue from we left of. Get epoch and date from the file name
            ## A sample model path might look like:
            ## /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.pth"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6))

        ## Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        ## Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "pod_pred_{}_*epoch*.pth".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{:04d}")

    def forward(self, x):
        c1, c2, c3, c4 = self.encoder(x)
        #384 torch.Size([2, 256, 96, 96]) torch.Size([2, 512, 48, 48]) torch.Size([2, 1024, 24, 24]) torch.Size([2, 2048, 12, 12])
        #416 torch.Size([2, 256, 104, 104]) torch.Size([2, 512, 52, 52]) torch.Size([2, 1024, 26, 26]) torch.Size([2, 2048, 13, 13])
        final = self.decoder1(c1, c2, c3, c4)
        return final
