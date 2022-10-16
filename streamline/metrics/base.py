from abc import ABC, abstractmethod
from ..datasets import DatasetFactory
from ..models import ModelFactory


import os
import torch


class ExperimentMetric(ABC):

    def __init__(self, db_loc, base_exp_directory, dataset_root_directory, gpu_name, batch_size, round_join_metric_tuple):
        
        # Instantiate every field of the metric primary key EXCEPT
        # the metric name, which should be instantiated by subtypes
        self.db_loc = db_loc
        self.base_exp_directory = base_exp_directory
        self.dataset_root_directory = dataset_root_directory
        self.gpu_name = gpu_name
        self.batch_size = batch_size

        # Unpack the tuple
        dataset_name, model_architecture_name, limited_memory, arrival_pattern, run_number,\
        training_loop, al_method, al_budget, init_task_size, unl_buffer_size, round_num, current_epoch, \
        dataset_split_path, model_weight_path, opt_state_path, lr_state_path, completed_unix_time, \
        metric_name, eval_dataset, value, computed_unix_time = round_join_metric_tuple

        # Store all but current epoch, completed unix time, metric name (handled by subclasses), value (to be computed), and time (to be computed)
        self.dataset_name = dataset_name
        self.model_architecture_name = model_architecture_name
        self.limited_mem = limited_memory
        self.arrival_pattern = arrival_pattern
        self.run_number = run_number
        self.training_loop = training_loop
        self.al_method = al_method
        self.al_budget = al_budget
        self.init_task_size = init_task_size
        self.unl_buffer_size = unl_buffer_size
        self.round_num = round_num
        self.dataset_split_path = dataset_split_path
        self.model_path = model_weight_path
        self.opt_state_path = opt_state_path
        self.lr_state_path = lr_state_path
        self.eval_dataset_name = eval_dataset


    def load_dataset(self):

        dataset_factory = DatasetFactory(self.dataset_root_directory)
        eval_dataset, eval_transform, num_classes = dataset_factory.get_eval_dataset(self.eval_dataset_name)
        return eval_dataset, eval_transform, num_classes


    def load_model(self, num_classes):

        pretrain_weight_directory = os.path.join(self.base_exp_directory, "weights")
        model_factory = ModelFactory(num_classes=num_classes, pretrain_weight_directory=pretrain_weight_directory)
        model = model_factory.get_model(self.model_architecture_name)

        model_path = os.path.join(self.base_exp_directory, "weights", self.model_path)
        with open(model_path, "rb") as f:
            model_state_dict = torch.load(f, map_location=torch.device("cpu"))

        model.load_state_dict(model_state_dict)
        return model


    @abstractmethod
    def evaluate(self):
        
        # Subtypes will calculate the value field
        pass