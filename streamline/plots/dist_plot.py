import matplotlib
from streamline import persistence
from streamline import datasets
from streamline.utils.markov import sample_random_access_chain, sample_sequential_access_chain
from .base import BasePlotter

import math
import matplotlib.pyplot as plt
import numpy as np


class DistPlotter(BasePlotter):

    def __init__(self, plotter_config):
        super(DistPlotter, self).__init__(plotter_config)
        self.set_defaults()
        self.line_mapping = {}


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


    def plot_column(self, current_axis, col_num):

        # Get the current axis and plot name
        plot_name       = F"plot_{col_num + 1}"
        plot_config     = self.plotter_config[plot_name]

        # Get the axis metric information
        x_axis_metric   = "al_round"
        y_axis_metric_b = "task_presence_"
        x_axis_label    = "AL Round"
        y_axis_label    = "Size of Slice"

        # Get the experiment information
        db_loc              = plot_config["db_loc"]
        train_dataset       = plot_config["train_dataset"]
        model_architecture  = plot_config["model_architecture"]
        limited_mem         = plot_config["limited_mem"]
        arrival_pattern     = plot_config["arrival_pattern"]
        colors              = plot_config["colors"]

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
        
        # Create x, y
        current_axis[0][col_num].set_title(r"\textbf{" + train_dataset + "}", fontsize=self.plotter_config["title_font_size"], fontfamily=self.plotter_config["font"])
        current_axis[0][col_num].set_xlabel(r"\textbf{" + x_axis_label + "}", fontsize=self.plotter_config["x_axis_font_size"], fontfamily=self.plotter_config["font"])
        current_axis[0][col_num].set_ylabel(r"\textbf{" + y_axis_label + " (Streamline)}", fontsize=self.plotter_config["y_axis_font_size"], fontfamily=self.plotter_config["font"])
        current_axis[1][col_num].set_xlabel(r"\textbf{" + x_axis_label + "}", fontsize=self.plotter_config["x_axis_font_size"], fontfamily=self.plotter_config["font"])
        current_axis[1][col_num].set_ylabel(r"\textbf{" + y_axis_label + " (Other)}", fontsize=self.plotter_config["y_axis_font_size"], fontfamily=self.plotter_config["font"])

        training_loop, al_method, al_budget, init_task_size, unl_buffer_size, eval_dataset, point_style, line_style, line_name = plot_config["experiments"][0]
        for task_number in range(num_tasks):
            y_axis_metric   = F"{y_axis_metric_b}{task_number}"
            color           = colors[task_number]

            # Get the metrics for the x axis and the y axis. Additionally, conform them to the same length.
            y_axis_avg, y_axis_std  = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                                    al_method, al_budget, init_task_size, unl_buffer_size, y_axis_metric, eval_dataset, 100)
            x_axis_avg, _           = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                                    al_method, al_budget, init_task_size, unl_buffer_size, x_axis_metric, eval_dataset, 100)

            # Set the xticks to be x_axis_avg if not streamline, which is the only method that will differ. Do every other value.
            if "streamline" not in al_method:
                current_axis[0][col_num].set_xticks([x for i,x in enumerate(x_axis_avg) if i % 2 == 0])

            conformed_length = min(y_axis_avg.shape[0], x_axis_avg.shape[0])
            x_axis_avg = x_axis_avg[:conformed_length]
            y_axis_avg = y_axis_avg[:conformed_length]
            y_axis_std = y_axis_std[:conformed_length]

            # Calculate y-metric fill arrays, which reflect 1 std away from the avg.
            y_metric_below = y_axis_avg - y_axis_std
            y_metric_above = y_axis_avg + y_axis_std

            line_name = r"\textsc{" + str(task_number) + r"}"


            line_for_legend = current_axis[0][col_num].plot(x_axis_avg, y_axis_avg, color=color, label=line_name, marker=point_style, linestyle=line_style)[0]
            current_axis[0][col_num].fill_between(x_axis_avg, y_metric_below, y_metric_above, color=color, label=line_name, alpha=0.15)

            if line_name not in self.line_mapping:
                self.line_mapping[line_name] = line_for_legend

        training_loop, al_method, al_budget, init_task_size, unl_buffer_size, eval_dataset, point_style, line_style, line_name = plot_config["experiments"][1]
        for task_number in range(num_tasks):
            y_axis_metric = F"{y_axis_metric_b}{task_number}"
            color           = colors[task_number]

            # Get the metrics for the x axis and the y axis. Additionally, conform them to the same length.
            y_axis_avg, y_axis_std  = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                                    al_method, al_budget, init_task_size, unl_buffer_size, y_axis_metric, eval_dataset, 100)
            x_axis_avg, _           = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                                    al_method, al_budget, init_task_size, unl_buffer_size, x_axis_metric, eval_dataset, 100)

            # Set the xticks to be x_axis_avg if not streamline, which is the only method that will differ. Do every other value.
            if "streamline" not in al_method:
                current_axis[1][col_num].set_xticks([x for i,x in enumerate(x_axis_avg) if i % 2 == 0])

            conformed_length = min(y_axis_avg.shape[0], x_axis_avg.shape[0])
            x_axis_avg = x_axis_avg[:conformed_length]
            y_axis_avg = y_axis_avg[:conformed_length]
            y_axis_std = y_axis_std[:conformed_length]

            # Calculate y-metric fill arrays, which reflect 1 std away from the avg.
            y_metric_below = y_axis_avg - y_axis_std
            y_metric_above = y_axis_avg + y_axis_std

            line_name = r"\textsc{" + str(task_number) + r"}"


            line_for_legend = current_axis[1][col_num].plot(x_axis_avg, y_axis_avg, color=color, label=line_name, marker=point_style, linestyle=line_style)[0]
            current_axis[1][col_num].fill_between(x_axis_avg, y_metric_below, y_metric_above, color=color, label=line_name, alpha=0.15)

            if line_name not in self.line_mapping:
                self.line_mapping[line_name] = line_for_legend

        # Place grid
        current_axis[0][col_num].grid(axis="y", linestyle="-", linewidth=1)
        current_axis[0][col_num].grid(axis="x", linestyle="-", linewidth=1)
        current_axis[1][col_num].grid(axis="y", linestyle="-", linewidth=1)
        current_axis[1][col_num].grid(axis="x", linestyle="-", linewidth=1)


    def produce_plot(self):

        # Create a base figure by determining how many columns are needed.
        num_rows = 2
        num_cols = -1
        for plot_attr in self.plotter_config:
            if plot_attr.startswith("plot_"):
                _, col_num = plot_attr.split("_")
                num_cols = max(int(col_num), num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=self.plotter_config["figsize"], squeeze=False, gridspec_kw=self.plotter_config["gridspec_kw"])

        # Plot each cell
        for col in range(1, num_cols + 1):
            if F"plot_{col}" in self.plotter_config:
                self.plot_column(axes, col - 1)

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