'''
Code for loading models trained with EigenPlaces as a global features extractor
for geolocalization through image retrieval.
Multiple models are available with different backbones. Below is a summary of
models available (backbone : list of available output descriptors
dimensionality). For example you can use a model based on a ResNet50 with
descriptors dimensionality 1024.
    ResNet18:  [32, 64, 128, 256, 512]
    ResNet50:  [32, 64, 128, 256, 512, 1024, 2048]
    ResNet101: [32, 64, 128, 256, 512, 1024, 2048]
    ResNet152: [32, 64, 128, 256, 512, 1024, 2048]
    VGG16:     [    64, 128, 256, 512]
Basically these configurations are the same as with CosPlace.

EigenPlaces paper: https://arxiv.org/abs/2308.10832
'''

import torch
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel


class EigenPlaces(BaseModel):
    # Same as CosPlace
    default_conf = {
        'backbone': 'ResNet50',
        'fc_output_dim' : 2048
    }
    required_inputs = ['image']
    def _init(self, conf):
        self.net = torch.hub.load(
            'gmberton/eigenplaces',
            'get_trained_model',
            backbone=conf['backbone'],
            fc_output_dim=conf['fc_output_dim']
        ).eval()
        
        # Same as CosPlace too, see EigenPlaces/train.py on https://github.com/gmberton/EigenPlaces
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.norm_rgb = tvf.Normalize(mean=mean, std=std)

    def _forward(self, data):
        image = self.norm_rgb(data['image'])
        desc = self.net(image)
        return {
            'global_descriptor': desc,
        }
