from abc import ABC, abstractmethod
from streamline import persistence

import numpy as np

class BasePlotter(ABC):

    def __init__(self, plotter_config):
        self.plotter_config = plotter_config


    def find_labels_needed_to_reach_target_metric_value(self, num_labels_list, metric_list, target_metric_value):

        # Assuming num_labels_list is sorted and that metric_list is paired with num_labels_list, find
        # the value in metric_list that is just higher than the target for interpolation. IF, however, 
        # the value is the same, return the label value instead.
        high_index = -1
        for i, metric in enumerate(metric_list):
            if abs(metric - target_metric_value) < 1e-7:
                return num_labels_list[i]
            elif metric >= target_metric_value:
                high_index = i
                break

        # If no value can be interpolated, then return None.
        if high_index == 0:
            return metric_list[0]

        # Now, we can approximate the labels needed to reach the target metric value by linearly interpolating
        # between the higher metric value and the metric value right below it. Doing so will allow us to determine
        # the labels needed.
        m                   = (num_labels_list[high_index] - num_labels_list[high_index - 1]) / (metric_list[high_index] - metric_list[high_index - 1])
        interpolated_labels = m * (target_metric_value - metric_list[high_index - 1]) + num_labels_list[high_index - 1]

        return interpolated_labels


    def calculate_labeling_efficiency_list(self, base_num_labels_list, base_metric_list, compared_num_labels_list, compared_metric_list, granularity=10):

        #print(base_num_labels_list, base_metric_list, compared_num_labels_list, compared_metric_list)

        # Calculate the comparison metric values, which is determined by the method with the smallest range.
        highest_overlapping_value = min(max(base_metric_list), max(compared_metric_list))
        lowest_overlapping_value  = max(min(base_metric_list), min(compared_metric_list))
        metric_range              = highest_overlapping_value - lowest_overlapping_value
        sample_metric_values      = [lowest_overlapping_value + (metric_range / granularity) * x for x in range(granularity + 1)]

        # Interpolate the number of labels needed to achieve the sample metric values for each strategy. Calculate their ratio as the labeling efficiency.
        labeling_efficiencies = []
        for sample_metric_value in sample_metric_values:
            interpolated_base_label_count     = self.find_labels_needed_to_reach_target_metric_value(base_num_labels_list, base_metric_list, sample_metric_value)
            interpolated_compared_label_count = self.find_labels_needed_to_reach_target_metric_value(compared_num_labels_list, compared_metric_list, sample_metric_value)

            if interpolated_compared_label_count is None:
                labeling_efficiencies.append(None)
                continue
            
            #print(sample_metric_value)
            #print(interpolated_base_label_count)
            #print(interpolated_compared_label_count)

            labeling_efficiency = interpolated_base_label_count / interpolated_compared_label_count
            labeling_efficiencies.append(labeling_efficiency)

        return sample_metric_values, labeling_efficiencies


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
            
            # Get the initial size of the dataset being examined in the training set.
            if train_dataset == "RotatedMNIST":
                num_tasks=5
            elif train_dataset == "OrganMNIST":
                num_tasks=3
            elif train_dataset == "PermutedMNIST":
                num_tasks=5
            elif train_dataset == "Office31":
                num_tasks=3
            elif train_dataset == "IWildCam":
                num_tasks=4
            elif train_dataset == "FMOW":
                num_tasks=3
            elif train_dataset == "PovertyMap":
                num_tasks=2
            elif train_dataset == "BDD100K":
                num_tasks=2
            elif train_dataset == "KITTIFog":
                num_tasks=2
            elif train_dataset == "CityscapesRain":
                num_tasks=2
            elif train_dataset == "CityscapesFog":
                num_tasks=2
            else:
                raise ValueError("Dataset not implemented!")
            initial_seed_size = init_task_size * (num_tasks - 1) + (init_task_size // 5)

            run_metric_array = np.cumsum(run_metric_array, axis=1) + initial_seed_size

        metric_avgs = np.average(run_metric_array, axis=0)
        metric_stds = np.zeros_like(metric_avgs) if run_metric_array.shape[0] == 1 else np.std(run_metric_array, axis=0, ddof=1)  # Specify delta degrees of freedom to be 1 for Bessel's correction

        return metric_avgs, metric_stds        


    @abstractmethod
    def produce_plot(self):
        pass