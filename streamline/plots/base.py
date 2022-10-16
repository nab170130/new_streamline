from abc import ABC, abstractmethod
from streamline import persistence

import numpy as np

class BasePlotter(ABC):

    def __init__(self, plotter_config):
        self.plotter_config = plotter_config


    def get_avg_std_metric_list(self, db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                al_method, al_budget, init_task_size, unl_buffer_size, metric, eval_dataset, max_plot_points):

        if metric == "al_round":
            metric_avgs = np.arange(max_plot_points)
            metric_stds = None
            return metric_avgs, metric_stds

        _, run_metric_value_lists, _ = persistence.get_valid_metric_values_computed_times_over_all_runs(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern,
                                                                                                        training_loop, al_method, al_budget, init_task_size, unl_buffer_size,
                                                                                                        metric, eval_dataset)
        
        if len(run_metric_value_lists) == 0:
            raise ValueError(F"No metric information for tuple {(train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, al_method, al_budget, init_task_size, unl_buffer_size, metric, eval_dataset)}")

        # The returned lists might not have all the same lengths. To compute avg/std, we shall truncate the longer ones
        # so that they conform to the smallest list. Additionally, we need the conformed length to be at most max_plot_points.
        conformed_length = min(max_plot_points, min([len(run_metric_value_list) for run_metric_value_list in run_metric_value_lists]))
        for list_idx, run_metric_value_list in enumerate(run_metric_value_lists):
            run_metric_value_lists[list_idx] = run_metric_value_list[:conformed_length]

        # Calculate avg/std via numpy. If the metric is labeled_instances, then we need to cumulate the metric first.
        run_metric_array = np.array(run_metric_value_lists)
        if metric == "labeled_instances":
            run_metric_array = np.cumsum(run_metric_array, axis=1)

        metric_avgs = np.average(run_metric_array, axis=0)
        metric_stds = np.zeros_like(metric_avgs) if run_metric_array.shape[0] == 1 else np.std(run_metric_array, axis=0, ddof=1)  # Specify delta degrees of freedom to be 1 for Bessel's correction

        return metric_avgs, metric_stds        


    @abstractmethod
    def produce_plot(self):
        pass