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

class PerTaskMeanAveragePrecisionMetric(ExperimentMetric):

    def __init__(self, db_loc, base_exp_directory, dataset_root_directory, gpu_name, batch_size, round_join_metric_tuple, task_num):
        
        super(PerTaskMeanAveragePrecisionMetric, self).__init__(db_loc, base_exp_directory, dataset_root_directory, gpu_name, batch_size, round_join_metric_tuple)
        self.task_num = task_num
        self.name = F"per_task_mAP_{task_num}"

    def evaluate(self):

        checkpoint_path     = os.path.join(self.base_exp_directory, "weights", self.model_path)
        abs_working_dir, _  = os.path.split(checkpoint_path)

        # BDD100K requires different treatment. We will use MMDetection to load the base dataset based on the configs given in the utils folder.
        bdd100k_config_path             = "streamline/utils/mmdet_configs/bdd100k/faster_rcnn_r50_fpn_1x_bdd100k_cocofmt.py"
        bdd100k_config                  = Config.fromfile(bdd100k_config_path)
        bdd100k_config['device']        = self.gpu_name.split(":")[0]
        bdd100k_config['gpu_ids']       = [int(self.gpu_name.split(":")[1])]
        bdd100k_config["work_dir"]      = abs_working_dir
        bdd100k_config["resume_from"]   = checkpoint_path
        bdd100k_config["seed"]      = 0

        # Get the evaluation dataset
        eval_dataset, eval_transform, num_classes = self.load_dataset()

        # Unlike in the full map calculation, we can achieve a per-task evaluation by simply
        # modifying the base mapping in the eval dataset.
        focused_partition                               = eval_dataset.task_idx_partitions[self.task_num]
        mapping_slice                                   = [eval_dataset.bdd100k_base_mapping[i] for i in focused_partition]
        eval_dataset.bdd100k_base_mapping               = mapping_slice
        eval_dataset.task_idx_partitions                = [[] for x in eval_dataset.task_idx_partitions]
        eval_dataset.task_idx_partitions[self.task_num] = list(range(len(focused_partition)))
            
        # Get the model from the checkpoint and a dataloader
        model       = init_detector(bdd100k_config, checkpoint=checkpoint_path, device=self.gpu_name, cfg_options=None)
        eval_loader = build_dataloader(eval_dataset, bdd100k_config["data"]["samples_per_gpu"], bdd100k_config["data"]["workers_per_gpu"], 
                                        num_gpus=1, dist=True, shuffle=False, seed=bdd100k_config["seed"])

        model       = build_dp(model, bdd100k_config["device"], device_ids=bdd100k_config["gpu_ids"])
        outputs     = single_gpu_test(model, eval_loader)

        eval_results = eval_dataset.evaluate(outputs)
        
        self.value = eval_results["bbox_mAP_50"]
        self.time = time.time_ns()