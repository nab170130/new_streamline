from streamline import persistence
from .base import BasePlotter

import math
import matplotlib.pyplot as plt
import numpy as np


class OnePlotter(BasePlotter):

    def __init__(self, plotter_config):
        super(OnePlotter, self).__init__(plotter_config)


    def produce_plot(self):
        
        # Create a base figure
        fig = plt.figure(figsize=self.plotter_config["figsize"])

        # Get the axis metric information
        x_axis_metric = self.plotter_config["x_axis_measure"]
        y_axis_metric = self.plotter_config["y_axis_measure"]

        # Get the experiment information
        db_loc              = self.plotter_config["db_loc"]
        train_dataset       = self.plotter_config["train_dataset"]
        model_architecture  = self.plotter_config["model_architecture"]
        limited_mem         = self.plotter_config["limited_mem"]
        arrival_pattern     = self.plotter_config["arrival_pattern"]
        max_plot_points     = self.plotter_config["max_plot_points"]
        
        # Create x, y, title
        plt.title(train_dataset, fontsize=self.plotter_config["title_font_size"], fontfamily=self.plotter_config["font"])
        plt.xlabel(x_axis_metric, fontsize=self.plotter_config["x_axis_font_size"], fontfamily=self.plotter_config["font"])
        plt.ylabel(y_axis_metric, fontsize=self.plotter_config["y_axis_font_size"], fontfamily=self.plotter_config["font"])

        line_mapping = {}
        for training_loop, al_method, al_budget, init_task_size, unl_buffer_size, eval_dataset, color in self.plotter_config["experiments"]:
            
            # Get the metrics for the x axis and the y axis. Additionally, conform them to the same length.
            y_axis_avg, y_axis_std  = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                                    al_method, al_budget, init_task_size, unl_buffer_size, y_axis_metric, eval_dataset, max_plot_points)
            x_axis_avg, _           = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                                    al_method, al_budget, init_task_size, unl_buffer_size, x_axis_metric, eval_dataset, max_plot_points)

            conformed_length = min(y_axis_avg.shape[0], x_axis_avg.shape[0])
            x_axis_avg = x_axis_avg[:conformed_length]
            y_axis_avg = y_axis_avg[:conformed_length]
            y_axis_std = y_axis_std[:conformed_length]

            # Calculate y-metric fill arrays, which reflect 1 std away from the avg.
            y_metric_below = y_axis_avg - y_axis_std
            y_metric_above = y_axis_avg + y_axis_std

            line_for_legend = plt.plot(x_axis_avg, y_axis_avg, color=color, label=al_method, marker="o")[0]
            plt.fill_between(x_axis_avg, y_metric_below, y_metric_above, color=color, label=al_method, alpha=0.3)

            if (training_loop, al_method) not in line_mapping:
                line_mapping[(training_loop, al_method)] = line_for_legend

        # Prepare the legend
        lines = []
        num_methods = len(line_mapping.keys())
        for training_loop, al_method in line_mapping:
            line = line_mapping[(training_loop, al_method)]
            lines.append(line)
        fig.legend(handles=lines, loc="upper center", ncol=num_methods, fontsize=self.plotter_config["legend_font_size"])

        return fig