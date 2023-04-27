from streamline import persistence
from .base import BasePlotter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class BarPlotter(BasePlotter):

    def __init__(self, plotter_config):
        super(BarPlotter, self).__init__(plotter_config)
        self.set_defaults()


    def get_labeling_efficiency_values(self, bar_config):

        # Get the experiment information
        db_loc              = bar_config["db_loc"]
        train_dataset       = bar_config["train_dataset"]
        model_architecture  = bar_config["model_architecture"]
        limited_mem         = bar_config["limited_mem"]
        arrival_pattern     = bar_config["arrival_pattern"]
        metric              = bar_config["measure"]

        # Get metrics for random, streamline, and other.
        training_loop, al_method, al_budget, init_task_size, unl_buffer_size, eval_dataset = bar_config["streamline"]
        streamline_metric_avg, _    = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                        al_method, al_budget, init_task_size, unl_buffer_size, metric, eval_dataset, max_plot_points=100)
        streamline_labels_avg, _    = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                        al_method, al_budget, init_task_size, unl_buffer_size, "labeled_instances", eval_dataset, max_plot_points=100)

        training_loop, al_method, al_budget, init_task_size, unl_buffer_size, eval_dataset = bar_config["other"]
        other_metric_avg, _ = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                        al_method, al_budget, init_task_size, unl_buffer_size, metric, eval_dataset, max_plot_points=100)
        other_labels_avg, _ = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                        al_method, al_budget, init_task_size, unl_buffer_size, "labeled_instances", eval_dataset, max_plot_points=100)

        random_metric_avg, _ = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                        "random", al_budget, init_task_size, unl_buffer_size, metric, eval_dataset, max_plot_points=100)
        random_labels_avg, _ = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                        "random", al_budget, init_task_size, unl_buffer_size, "labeled_instances", eval_dataset, max_plot_points=100)

        # Look at the final metrics in each list and determine the smallest of them
        smallest_target_metric_value = min(streamline_metric_avg[-1], other_metric_avg[-1], random_metric_avg[-1])

        # Compute the labeling efficiency of streamline and the other method at this value
        random_labels_needed        = self.find_labels_needed_to_reach_target_metric_value(random_labels_avg, random_metric_avg, smallest_target_metric_value)
        streamline_labels_needed    = self.find_labels_needed_to_reach_target_metric_value(streamline_labels_avg, streamline_metric_avg, smallest_target_metric_value)
        other_labels_needed         = self.find_labels_needed_to_reach_target_metric_value(other_labels_avg, other_metric_avg, smallest_target_metric_value)

        streamline_efficiency   = random_labels_needed / streamline_labels_needed
        other_efficiency        = random_labels_needed / other_labels_needed

        return smallest_target_metric_value, streamline_efficiency, other_efficiency


    def produce_plot(self):
        
        # Create a base figure
        fig = plt.figure(figsize=self.plotter_config["figsize"])

        # Group streamline final values and competitor final values.
        num_bars = -1
        for plot_attr in self.plotter_config:
            if plot_attr.startswith("bar_") and plot_attr != "bar_width":
                _, bar_num = plot_attr.split("_")
                num_bars = max(int(bar_num), num_bars)

        met_vals        = [0. for x in range(num_bars)]
        streamline_vals = [0. for x in range(num_bars)]
        other_vals      = [0. for x in range(num_bars)]
        dataset_labels  = ["" for x in range(num_bars)]

        for key in self.plotter_config:
            if key.startswith("bar_") and key != "bar_width":
                bar_number = int(key.split("_")[1]) - 1
                target_metric_value, streamline_efficiency, other_efficiency    = self.get_labeling_efficiency_values(self.plotter_config[key])
                met_vals[bar_number]                                            = round(target_metric_value, 2)
                streamline_vals[bar_number]                                     = streamline_efficiency
                other_vals[bar_number]                                          = other_efficiency
                dataset_labels[bar_number]                                      = self.plotter_config[key]["train_dataset"]
                
        # Finish the plot: img cls
        x           = np.arange(len(streamline_vals))
        bar_width   = self.plotter_config["bar_width"]
        bars = plt.bar(x - bar_width/2, streamline_vals, bar_width, label=r"\textsc{Streamline}", zorder=3, color=(0, 0.6, 0))
        plt.bar(x + bar_width/2, other_vals, bar_width, label=r"\textsc{Next Best Baseline}", zorder=3, color=(0.8, 0, 0))
        plt.ylabel(r"\textbf{Labeling Efficiency}", fontsize=self.plotter_config["y_axis_font_size"], fontfamily=self.plotter_config["font"])
        plt.xticks(x, [r"\textbf{" + lab + "}\n(" + r"\textbf{" + str(val) + "})" for lab, val in zip(dataset_labels, met_vals)], fontsize=20)
        plt.legend(fontsize=self.plotter_config["legend_font_size"])
        plt.grid(axis="y", linestyle="-", linewidth=1, zorder=0)
        plt.tight_layout()

        return fig

    
    def set_defaults(self):

        matplotlib.rcParams['text.usetex'] = True
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        matplotlib.rc('text', usetex=True)
        plt.rc('axes', linewidth=1)
        plt.rc('font', weight='bold')