from .base import ExperimentMetric
from ..datasets import TransformSubset

from torch.utils.data import DataLoader

import time
import torch

class PerTaskAccuracyMetric(ExperimentMetric):

    def __init__(self, db_loc, base_exp_directory, dataset_root_directory, gpu_name, batch_size, round_join_metric_tuple, task_num):
        
        super(PerTaskAccuracyMetric, self).__init__(db_loc, base_exp_directory, dataset_root_directory, gpu_name, batch_size, round_join_metric_tuple)
        self.task_num = task_num
        self.name = F"per_task_accuracy_{self.task_num}"


    def evaluate(self):

        # Get the evaluation dataset and the model
        eval_dataset, _, num_classes = self.load_dataset()
        model = self.load_model(num_classes)

        # Create a subset of the eval_dataset that corresponds only to the given task.
        idx_to_eval = eval_dataset.task_idx_partitions[self.task_num]
        new_task_idx_partitions = [[] for x in range(len(eval_dataset.task_idx_partitions))]
        new_task_idx_partitions[self.task_num] = idx_to_eval
        per_task_eval_dataset = TransformSubset(eval_dataset, new_task_idx_partitions)

        # Create a DataLoader and get class predictions
        eval_dataloader = DataLoader(per_task_eval_dataset, batch_size=self.batch_size, shuffle=True)

        total_correct = 0.
        with torch.no_grad():
            model = model.to(self.gpu_name)
            for data, labels in eval_dataloader:
                data, labels = data.to(self.gpu_name), labels.to(self.gpu_name)
                out = model(data)
                class_predictions = torch.max(out,1)[1]
                correct_vector = class_predictions == labels
                total_correct += torch.sum(1.0 * correct_vector).item()

        accuracy_on_eval_dataset = total_correct / len(per_task_eval_dataset)
        self.value = accuracy_on_eval_dataset
        self.time = time.time_ns()