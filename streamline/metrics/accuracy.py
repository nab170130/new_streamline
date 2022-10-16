from .base import ExperimentMetric

from torch.utils.data import DataLoader

import time
import torch

class AccuracyMetric(ExperimentMetric):

    def __init__(self, db_loc, base_exp_directory, dataset_root_directory, gpu_name, batch_size, round_join_metric_tuple):
        
        super(AccuracyMetric, self).__init__(db_loc, base_exp_directory, dataset_root_directory, gpu_name, batch_size, round_join_metric_tuple)
        self.name = "accuracy"


    def evaluate(self):

        # Get the evaluation dataset and the model
        eval_dataset, _, num_classes = self.load_dataset()
        model = self.load_model(num_classes)

        # Create a DataLoader and get class predictions
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=True)

        total_correct = 0.
        with torch.no_grad():
            model = model.to(self.gpu_name)
            for data, labels in eval_dataloader:
                data, labels = data.to(self.gpu_name), labels.to(self.gpu_name)
                out = model(data)
                class_predictions = torch.max(out,1)[1]
                correct_vector = class_predictions == labels
                total_correct += torch.sum(1.0 * correct_vector).item()

        accuracy_on_eval_dataset = total_correct / len(eval_dataset)
        self.value = accuracy_on_eval_dataset
        self.time = time.time_ns()