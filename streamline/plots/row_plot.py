import matplotlib
from streamline import persistence
from streamline import datasets
from streamline.utils.markov import sample_random_access_chain, sample_sequential_access_chain
from .base import BasePlotter

import math
import matplotlib.pyplot as plt
import numpy as np


class RowPlotter(BasePlotter):

    def __init__(self, plotter_config):
        super(RowPlotter, self).__init__(plotter_config)
        self.line_mapping = {}
        self.set_defaults()


    def plot_one_cell(self, current_axis, row_num, col_num):

        # Get the current axis and plot name
        plot_name       = F"plot_{row_num + 1}_{col_num + 1}"
        plot_config     = self.plotter_config[plot_name]

        # Get the axis metric information
        x_axis_metric   = plot_config["x_axis_measure"]
        y_axis_metric   = plot_config["y_axis_measure"]
        x_axis_label    = plot_config["x_axis_title"]
        y_axis_label    = plot_config["y_axis_title"]

        # Get the experiment information
        db_loc              = plot_config["db_loc"]
        train_dataset       = plot_config["train_dataset"]
        model_architecture  = plot_config["model_architecture"]
        limited_mem         = plot_config["limited_mem"]
        arrival_pattern     = plot_config["arrival_pattern"]
        max_plot_points     = plot_config["max_plot_points"]
        
        # Create x, y
        if row_num == 0: current_axis[row_num][col_num].set_title(r"\textbf{" + train_dataset + "}", fontsize=self.plotter_config["title_font_size"], fontfamily=self.plotter_config["font"])
        current_axis[row_num][col_num].set_xlabel(r"\textbf{" + x_axis_label + "}", fontsize=self.plotter_config["x_axis_font_size"], fontfamily=self.plotter_config["font"])
        current_axis[row_num][col_num].set_ylabel(r"\textbf{" + y_axis_label + "}", fontsize=self.plotter_config["y_axis_font_size"], fontfamily=self.plotter_config["font"])

        # If the labeling efficiency is being plotted, preprocess the labels and x-axis metric before attempting to plot.
        if y_axis_metric == "labeling_efficiency":
        
            # Get random, the base method to compare.
            for training_loop, al_method, al_budget, init_task_size, unl_buffer_size, eval_dataset, color, point_style, line_style, line_name in plot_config["experiments"]:
                if al_method == "random":
                    base_metric_values, base_metric_stds = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                                                        al_method, al_budget, init_task_size, unl_buffer_size, x_axis_metric, eval_dataset, max_plot_points)
                    base_labels_used, _ = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                                        al_method, al_budget, init_task_size, unl_buffer_size, "labeled_instances", eval_dataset, max_plot_points)

            # Now, for each thing to plot, compute labeling efficiencies.
            for training_loop, al_method, al_budget, init_task_size, unl_buffer_size, eval_dataset, color, point_style, line_style, line_name in plot_config["experiments"]:
                
                compared_metric_values, compared_metric_stds = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                                                            al_method, al_budget, init_task_size, unl_buffer_size, x_axis_metric, eval_dataset, max_plot_points)
                compared_labels_used, _ = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                                        al_method, al_budget, init_task_size, unl_buffer_size, "labeled_instances", eval_dataset, max_plot_points)

                x_axis_avg, y_axis_avg = self.calculate_labeling_efficiency_list(base_labels_used, base_metric_values, compared_labels_used, compared_metric_values)
                x_axis_avg = np.array(x_axis_avg)
                y_axis_avg = np.array(y_axis_avg)

                # Set the xticks to be x_axis_avg if not streamline, which is the only method that will differ. Do every other value.
                if al_method == "random":
                    current_axis[row_num][col_num].set_xticks([round(x, 2) for i,x in enumerate(x_axis_avg) if i % 3 == 0])

                conformed_length = min(y_axis_avg.shape[0], x_axis_avg.shape[0])
                x_axis_avg = x_axis_avg[:conformed_length]
                y_axis_avg = y_axis_avg[:conformed_length]

                line_name = r"\textsc{" + line_name + r"}"
                
                line_for_legend = current_axis[row_num][col_num].plot(x_axis_avg, y_axis_avg, color=color, label=line_name, marker=point_style, linestyle=line_style)[0]
                
                if line_name not in self.line_mapping:
                    self.line_mapping[line_name] = line_for_legend

        else:
            for training_loop, al_method, al_budget, init_task_size, unl_buffer_size, eval_dataset, color, point_style, line_style, line_name in plot_config["experiments"]:
                
                # Get the metrics for the x axis and the y axis. Additionally, conform them to the same length.
                y_axis_avg, y_axis_std  = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                                        al_method, al_budget, init_task_size, unl_buffer_size, y_axis_metric, eval_dataset, max_plot_points)
                x_axis_avg, _           = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                                        al_method, al_budget, init_task_size, unl_buffer_size, x_axis_metric, eval_dataset, max_plot_points)

                # Set the xticks to be x_axis_avg if not streamline, which is the only method that will differ. Do every other value.
                if "streamline" not in al_method:
                    current_axis[row_num][col_num].set_xticks([x for i,x in enumerate(x_axis_avg) if i % 2 == 0])

                conformed_length = min(y_axis_avg.shape[0], x_axis_avg.shape[0])
                x_axis_avg = x_axis_avg[:conformed_length]
                y_axis_avg = y_axis_avg[:conformed_length]
                y_axis_std = y_axis_std[:conformed_length]

                # Calculate y-metric fill arrays, which reflect 1 std away from the avg.
                y_metric_below = y_axis_avg - y_axis_std
                y_metric_above = y_axis_avg + y_axis_std

                line_name = r"\textsc{" + line_name + r"}"

                if "streamline" in al_method:
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
                    else:
                        raise ValueError("Dataset not implemented!")
                    num_rounds = len(x_axis_avg)

                    if arrival_pattern == "sequential":
                        task_arrival_pattern = sample_sequential_access_chain(num_tasks, num_rounds)
                    elif arrival_pattern.startswith("rare_every"):
                        every_mod = int(arrival_pattern.split("_")[2])
                        task_arrival_pattern = sample_random_access_chain(num_tasks - 1, num_rounds)
                        start_idx = 1 + every_mod
                        while start_idx < len(task_arrival_pattern):
                            task_arrival_pattern[start_idx] = num_tasks - 1
                            start_idx = start_idx + every_mod
                            
                    for round_number in range(num_rounds - 1):
                        if task_arrival_pattern[round_number + 1] == num_tasks - 1:
                            line_for_legend = current_axis[row_num][col_num].plot(x_axis_avg[round_number:round_number+2], y_axis_avg[round_number:round_number+2], color=color, label=line_name, marker=point_style, linestyle="--")[0]
                        else:
                            line_for_legend = current_axis[row_num][col_num].plot(x_axis_avg[round_number:round_number+2], y_axis_avg[round_number:round_number+2], color=color, label=line_name, marker=point_style, linestyle=line_style)[0]
                else:

                    line_for_legend = current_axis[row_num][col_num].plot(x_axis_avg, y_axis_avg, color=color, label=line_name, marker=point_style, linestyle=line_style)[0]
                #current_axis[row_num][col_num].fill_between(x_axis_avg, y_metric_below, y_metric_above, color=color, label=line_name, alpha=0.3)

                if line_name not in self.line_mapping:
                    self.line_mapping[line_name] = line_for_legend

        # Place grid
        current_axis[row_num][col_num].grid(axis="y", linestyle="-", linewidth=1)
        current_axis[row_num][col_num].grid(axis="x", linestyle="-", linewidth=1)


    def set_defaults(self):

        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['axes.spines.right'] = False
        matplotlib.rcParams['axes.spines.top'] = False
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        matplotlib.rc('text', usetex=True)
        plt.rc('axes', linewidth=1)
        plt.rc('font', weight='bold')


    def produce_plot(self):

        # Create a base figure by determining how many cells are needed.
        num_rows = -1
        num_cols = -1
        for plot_attr in self.plotter_config:
            if plot_attr.startswith("plot_"):
                _, row_num, col_num = plot_attr.split("_")
                num_rows = max(int(row_num), num_rows)
                num_cols = max(int(col_num), num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=self.plotter_config["figsize"], squeeze=False, gridspec_kw=self.plotter_config["gridspec_kw"])

        # Plot each cell
        for row in range(1, num_rows + 1):
            for col in range(1, num_cols + 1):
                if F"plot_{row}_{col}" in self.plotter_config:
                    self.plot_one_cell(axes, row - 1, col - 1)

        # Prepare the legend
        lines = []
        num_methods = len(self.line_mapping.keys())
        for line_name in self.line_mapping:
            line = self.line_mapping[line_name]
            lines.append(line)
        if "legend_col" in self.plotter_config:
            col_in_legend = self.plotter_config["legend_col"]
        else:
            col_in_legend = num_methods
        fig.legend(handles=lines, loc="upper center", ncol=col_in_legend, fontsize=self.plotter_config["legend_font_size"])

        return fig

        