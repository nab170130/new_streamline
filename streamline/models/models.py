from .conv_x import Conv2, Conv4, Conv6, Conv8
from .densenet import DenseNet161
from .mnist_nets import MnistNet, DeeperMnistNet
from .resnet import resnet18, resnet34, resnet50
from .smaller_resnet import resnet20, resnet32, resnet44, resnet56, resnet110

from mmcv import Config
from mmdet.models import build_detector

import torch
import torch.nn as nn

class ModelFactory:
    
    def __init__(self, num_classes=1000, pretrain_weight_directory=None, obj_det_config_path=None):
        self.num_classes = num_classes
        self.pretrain_weight_directory = pretrain_weight_directory
        self.obj_det_config_path = obj_det_config_path

    def get_model(self, model_name):
        
        if model_name.endswith("_1c"):
            channels = 1
        else:
            channels = 3

        if model_name.startswith("resnet18"):
            return resnet18(num_classes=self.num_classes, channels=channels)
        elif model_name.startswith("resnet20"):
            return resnet20(num_classes=self.num_classes, channels=channels)
        elif model_name.startswith("resnet32"):
            return resnet32(num_classes=self.num_classes, channels=channels)
        elif model_name.startswith("resnet34"):
            return resnet34(num_classes=self.num_classes, channels=channels)
        elif model_name.startswith("resnet44"):
            return resnet44(num_classes=self.num_classes, channels=channels)
        elif model_name.startswith("resnet50"):
            return resnet50(num_classes=self.num_classes, channels=channels)
        elif model_name.startswith("resnet56"):
            return resnet56(num_classes=self.num_classes, channels=channels)
        elif model_name.startswith("resnet110"):
            return resnet110(num_classes=self.num_classes, channels=channels)
        elif model_name.startswith("conv2"):
            return Conv2(num_classes=self.num_classes, channels=channels)
        elif model_name.startswith("conv4"):
            return Conv4(num_classes=self.num_classes, channels=channels)
        elif model_name.startswith("conv6"):
            return Conv6(num_classes=self.num_classes, channels=channels)
        elif model_name.startswith("conv8"):
            return Conv8(num_classes=self.num_classes, channels=channels)
        elif model_name.startswith("mnistnet"):
            return MnistNet(self.num_classes)
        elif model_name.startswith("deepermnistnet"):
            return DeeperMnistNet(self.num_classes)
        elif model_name.startswith("densenet161"):
            return DenseNet161(pretrain_weight_directory=self.pretrain_weight_directory, num_classes=self.num_classes)
        elif model_name.startswith("faster_rcnn"):
            # Use MMDetection to load and construct the model.
            obj_det_config                                                  = Config.fromfile(self.obj_det_config_path)
            obj_det_config['model']['roi_head']['bbox_head']['num_classes'] = self.num_classes
            model               = build_detector(obj_det_config.model, train_cfg=obj_det_config.get('train_cfg'), test_cfg=obj_det_config.get('test_cfg'))
            model.init_weights()
            return model