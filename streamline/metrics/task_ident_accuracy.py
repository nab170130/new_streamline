import json
from os import access
from streamline.persistence.sql import get_all_rounds_from_run
from streamline.persistence.utils import get_absolute_paths
from streamline.utils.markov import sample_random_access_chain, sample_rare_access_chain, sample_sequential_access_chain
from .base import ExperimentMetric

from torch.utils.data import DataLoader

import time
import torch

class TaskIdentificationAccuracyMetric(ExperimentMetric):

    def __init__(self, db_loc, base_exp_directory, dataset_root_directory, gpu_name, batch_size, round_join_metric_tuple):
        
        super(TaskIdentificationAccuracyMetric, self).__init__(db_loc, base_exp_directory, dataset_root_directory, gpu_name, batch_size, round_join_metric_tuple)
        self.name = "task_identification_accuracy"

    def evaluate(self):

        # To figure out whether the right task was identified, we can see which task has different labeled instances
        # between this round and the last. If that task matches the one in the (static) access pattern, then the
        # task was correctly identified. Otherwise, it was not.

        # If this is the initial round, then there was no task to identify. Mark as "successfully" identifying the task (1).
        if self.round_num == 0:
            self.value = 1
            self.time = time.time_ns()
        
        else:

            # Get all round info associated with this run
            all_al_rounds = get_all_rounds_from_run(self.db_loc, self.dataset_name, self.model_architecture_name, self.limited_mem, self.arrival_pattern, self.run_number, 
                                                    self.training_loop, self.al_method, self.al_budget, self.init_task_size, self.unl_buffer_size)
            num_rounds = len(all_al_rounds)

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

            # Compare differences in training splits for each task. If there is a difference, then the current task number is the identified task.
            for task_number, (prev_train_labeled_task_idx, curr_train_labeled_task_idx) in enumerate(zip(prev_train_unlabeled_dataset_split["train"],
                                                                                                       curr_train_unlabeled_dataset_split["train"])):
                if set(curr_train_labeled_task_idx) != set(prev_train_labeled_task_idx):
                    actual_task_identity = task_number

            # Sample the access chain corresponding to this run.
            num_tasks = len(curr_train_unlabeled_dataset_split["train"])
            if self.arrival_pattern == "sequential":
                task_arrival_pattern = sample_sequential_access_chain(num_tasks, num_rounds)
            elif self.arrival_pattern == "rare_beginning":
                task_arrival_pattern = sample_random_access_chain(num_tasks - 1, num_rounds)
                task_arrival_pattern[1]     = num_tasks - 1
                task_arrival_pattern[9]     = num_tasks - 1
            else:
                raise ValueError("Unknown arrival pattern")

            print("MY CHAIN IS", task_arrival_pattern)

            # Finally, compare the identified task to what it should be according to the chain.
            expected_task_identity = task_arrival_pattern[self.round_num]
            self.value = 1. if expected_task_identity == actual_task_identity else 0.
            self.time = time.time_ns()