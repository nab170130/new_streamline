from .accuracy import AccuracyMetric
from .map import MeanAveragePrecisionMetric
from .labeled_instances import LabeledInstancesMetric
from .per_task_accuracy import PerTaskAccuracyMetric
from .per_task_map import PerTaskMeanAveragePrecisionMetric
from .task_ident_accuracy import TaskIdentificationAccuracyMetric
from .task_presence import TaskPresenceMetric

class MetricFactory:

    def __init__(self, db_loc, base_exp_directory, base_dataset_directory, gpu_to_assign, batch_size, round_join_metric_tuple, obj_det_config_path=None):

        self.db_loc = db_loc
        self.base_exp_directory = base_exp_directory
        self.dataset_root_directory = base_dataset_directory
        self.gpu_name = gpu_to_assign
        self.batch_size = batch_size
        self.round_join_metric_tuple = round_join_metric_tuple
        self.obj_det_config_path = obj_det_config_path


    def get_metric(self, metric_name):

        if metric_name == "accuracy":
            metric = AccuracyMetric(self.db_loc, self.base_exp_directory, self.dataset_root_directory, self.gpu_name, self.batch_size, self.round_join_metric_tuple)
        elif metric_name.startswith("per_task_accuracy"):
            task_number = int(metric_name.split("_")[3])
            metric = PerTaskAccuracyMetric(self.db_loc, self.base_exp_directory, self.dataset_root_directory, self.gpu_name, self.batch_size, self.round_join_metric_tuple, task_number)            
        elif metric_name == "labeled_instances":
            metric = LabeledInstancesMetric(self.db_loc, self.base_exp_directory, self.dataset_root_directory, self.gpu_name, self.batch_size, self.round_join_metric_tuple)
        elif metric_name == "task_identification_accuracy":
            metric = TaskIdentificationAccuracyMetric(self.db_loc, self.base_exp_directory, self.dataset_root_directory, self.gpu_name, self.batch_size, self.round_join_metric_tuple)
        elif metric_name.startswith("task_presence"):
            task_number = int(metric_name.split("_")[2])
            metric = TaskPresenceMetric(self.db_loc, self.base_exp_directory, self.dataset_root_directory, self.gpu_name, self.batch_size, self.round_join_metric_tuple, task_number)
        elif metric_name == "mAP":
            metric = MeanAveragePrecisionMetric(self.db_loc, self.base_exp_directory, self.dataset_root_directory, self.gpu_name, self.batch_size, self.round_join_metric_tuple, self.obj_det_config_path)
        elif metric_name.startswith("per_task_mAP"):
            task_number = int(metric_name.split("_")[3])
            metric = PerTaskMeanAveragePrecisionMetric(self.db_loc, self.base_exp_directory, self.dataset_root_directory, self.gpu_name, self.batch_size, self.round_join_metric_tuple, task_number, self.obj_det_config_path)
        else:
            raise ValueError(F"Metric name {metric_name} is not supported")

        # If execution reaches here without fault, go ahead and declare the metric's name to be that of the passed argument
        metric.name = metric_name
        return metric