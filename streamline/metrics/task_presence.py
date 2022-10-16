from streamline.persistence import get_absolute_paths
from .base import ExperimentMetric
from ..datasets import DatasetFactory

from torch.utils.data import DataLoader

import json
import time
import torch

class TaskPresenceMetric(ExperimentMetric):

    def __init__(self, db_loc, base_exp_directory, dataset_root_directory, gpu_name, batch_size, round_join_metric_tuple, task_num):
        
        super(TaskPresenceMetric, self).__init__(db_loc, base_exp_directory, dataset_root_directory, gpu_name, batch_size, round_join_metric_tuple)
        self.name = F"task_presence_{task_num}"
        self.task_num = task_num


    def evaluate(self):

        dataset_factory = DatasetFactory(self.base_exp_directory)

        # Get the previous round's split location
        current_round_split_path    = self.dataset_split_path
        abs_split_path, _, _, _ = get_absolute_paths(self.base_exp_directory, current_round_split_path, "", "", "")

        # Retrieve the split data
        with open(abs_split_path, "r") as f:
            train_unlabeled_dataset_split = json.load(f)

        # Get the number of elements in the training task split that belongs to the specified task.
        num_elements_in_task = len(train_unlabeled_dataset_split["train"][self.task_num])
        self.value = num_elements_in_task
        self.time = time.time_ns()