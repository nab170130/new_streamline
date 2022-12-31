from .base import ExperimentMetric
from ..persistence import generate_save_locations_for_al_round_det, get_absolute_paths_det

from torch.utils.data import DataLoader

from mmcv import Config
from mmdet.apis import single_gpu_test, init_detector
from mmdet.datasets import build_dataloader
from mmdet.utils import build_dp

import json
import os
import time
import torch

class MeanAveragePrecisionMetric(ExperimentMetric):

    def __init__(self, db_loc, base_exp_directory, dataset_root_directory, gpu_name, batch_size, round_join_metric_tuple, obj_det_config_path):
        
        super(MeanAveragePrecisionMetric, self).__init__(db_loc, base_exp_directory, dataset_root_directory, gpu_name, batch_size, round_join_metric_tuple)
        self.name = "mAP"
        self.obj_det_config_path = obj_det_config_path

    def evaluate(self):

        checkpoint_path     = os.path.join(self.base_exp_directory, "weights", self.model_path)
        abs_working_dir, _  = os.path.split(checkpoint_path)

        # BDD100K requires different treatment. We will use MMDetection to load the base dataset based on the configs given in the utils folder.
        obj_det_config                  = Config.fromfile(self.obj_det_config_path)
        obj_det_config['device']        = self.gpu_name.split(":")[0]
        obj_det_config['gpu_ids']       = [int(self.gpu_name.split(":")[1])]
        obj_det_config["work_dir"]      = abs_working_dir
        obj_det_config["resume_from"]   = checkpoint_path
        obj_det_config["seed"]      = 0

        # Get the evaluation dataset
        eval_dataset, eval_transform, num_classes = self.load_dataset()
        obj_det_config['model']['roi_head']['bbox_head']['num_classes'] = num_classes
            
        # Get the model from the checkpoint and a dataloader
        model       = init_detector(obj_det_config, checkpoint=checkpoint_path, device=self.gpu_name, cfg_options=None)
        eval_loader = build_dataloader(eval_dataset, obj_det_config["data"]["samples_per_gpu"], obj_det_config["data"]["workers_per_gpu"], 
                                        num_gpus=1, dist=True, shuffle=False, seed=obj_det_config["seed"])

        model       = build_dp(model, obj_det_config["device"], device_ids=obj_det_config["gpu_ids"])
        outputs     = single_gpu_test(model, eval_loader)

        eval_results = eval_dataset.evaluate(outputs)
        
        self.value = eval_results["bbox_mAP_50"]
        self.time = time.time_ns()