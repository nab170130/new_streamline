from streamline.persistence import get_absolute_paths, get_all_rounds_from_run
from .base import ExperimentMetric

from torch.utils.data import DataLoader

import json
import time
import torch

class LabeledInstancesMetric(ExperimentMetric):

    def __init__(self, db_loc, base_exp_directory, dataset_root_directory, gpu_name, batch_size, round_join_metric_tuple):
        
        super(LabeledInstancesMetric, self).__init__(db_loc, base_exp_directory, dataset_root_directory, gpu_name, batch_size, round_join_metric_tuple)
        self.name = "labeled_instances"


    def evaluate(self):

        # If this is the initial round, then there are no labels assigned. Set this value to 0 and return.
        # If this isn't the initial round, then load the index splits from this round and the previous round
        # and calculate the set difference.
        if self.round_num == 0:
            self.value = 0
            self.time = time.time_ns()
        
        else:

            # Get all round info associated with this run
            all_al_rounds = get_all_rounds_from_run(self.db_loc, self.dataset_name, self.model_architecture_name, self.limited_mem, self.arrival_pattern, self.run_number, 
                                                    self.training_loop, self.al_method, self.al_budget, self.init_task_size, self.unl_buffer_size)

            # Get the previous round's split location and this round's split location
            split_field_index           = 2
            previous_round_split_path   = all_al_rounds[self.round_num - 1][split_field_index]
            current_round_split_path    = self.dataset_split_path

            # Get absolute locations from which to load
            prev_abs_split_path, _, _, _ = get_absolute_paths(self.base_exp_directory, previous_round_split_path, "", "", "")
            curr_abs_split_path, _, _, _ = get_absolute_paths(self.base_exp_directory, current_round_split_path, "", "", "")

            # Retrieve the split data
            with open(prev_abs_split_path, "r") as f:
                prev_train_unlabeled_dataset_split = json.load(f)

            with open(curr_abs_split_path, "r") as f:
                curr_train_unlabeled_dataset_split = json.load(f)

            # Compare differences in training splits. Those new indices in the current split
            # that are not in the previous split are those that are just newly labeled.
            # Record that value as the value for this metric.
            prev_train_indices = []
            curr_train_indices = []

            for task_idx_partition in prev_train_unlabeled_dataset_split["train"]:
                prev_train_indices.extend(task_idx_partition)

            for task_idx_partition in curr_train_unlabeled_dataset_split["train"]:
                curr_train_indices.extend(task_idx_partition)

            if self.limited_mem:
                num_new_labeled_instances = len(set(curr_train_indices) - set(prev_train_indices))
            else:
                num_new_labeled_instances = len(curr_train_indices) - len(prev_train_indices)
            self.value = num_new_labeled_instances
            self.time = time.time_ns()