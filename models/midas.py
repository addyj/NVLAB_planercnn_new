import torch
import torch.nn as nn

from .midas_blocks import FeatureFusionBlock, Interpolate, _make_encoder, _make_decoder

class Encoder(nn.Module):
    def __init__(self, state_dict):
        super(Encoder, self).__init__()

        self.use_pretrained = True
        self.pretrained = _make_encoder(self.use_pretrained)
        self.parameters = state_dict
        self.load_state_dict(self.parameters, strict=False)

    def forward(self, x):

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        return layer_1, layer_2, layer_3, layer_4

class Decoder1(nn.Module):
    def __init__(self, state_dict):
        super(Decoder1, self).__init__()

        self.scratch = _make_decoder(256)
        self.scratch.refinenet4 = FeatureFusionBlock(256)
        self.scratch.refinenet3 = FeatureFusionBlock(256)
        self.scratch.refinenet2 = FeatureFusionBlock(256)
        self.scratch.refinenet1 = FeatureFusionBlock(256)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
        )

        self.parameters = state_dict
        self.load_state_dict(self.parameters, strict=False)


    def forward(self, x1, x2, x3, x4):

        layer_1_rn = self.scratch.layer1_rn(x1)
        layer_2_rn = self.scratch.layer2_rn(x2)
        layer_3_rn = self.scratch.layer3_rn(x3)
        layer_4_rn = self.scratch.layer4_rn(x4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return torch.squeeze(out, dim=1)
