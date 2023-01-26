from streamline import persistence
from streamline import datasets
from streamline.utils.markov import sample_random_access_chain, sample_sequential_access_chain
from .base import BasePlotter

import math
import numpy as np


class STDTable(BasePlotter):

    def __init__(self, plotter_config):
        super(STDTable, self).__init__(plotter_config)


    def generate_table(self, table_name):

        # Get the current axis and plot name
        plot_config     = self.plotter_config[table_name]

        # Get metric information
        metric   = plot_config["measure"]

        # Get the experiment information
        db_loc              = plot_config["db_loc"]
        train_dataset       = plot_config["train_dataset"]
        model_architecture  = plot_config["model_architecture"]
        limited_mem         = plot_config["limited_mem"]
        arrival_pattern     = plot_config["arrival_pattern"]
        
        # Build a string to copy to LaTeX
        table_string = ""

        for training_loop, al_method, al_budget, init_task_size, unl_buffer_size, eval_dataset, line_name in plot_config["experiments"]:
            
            # Get the metrics for the x axis and the y axis. Additionally, conform them to the same length.
            y_axis_avg, y_axis_std  = self.get_avg_std_metric_list(db_loc, train_dataset, model_architecture, limited_mem, arrival_pattern, training_loop, 
                                                                    al_method, al_budget, init_task_size, unl_buffer_size, metric, eval_dataset, 100)

            if table_string == "":
                table_string    = r"\begin{tabular}{|"
                num_rounds      = len(y_axis_std)
                table_string    = table_string + "c|" * (num_rounds + 1) + "}\n" + r"\hline" + "\n"
                table_string    = table_string + "\t" + r"\textbf{" + train_dataset + "} "
                for i in range(num_rounds):
                    table_string = table_string + " & " + str(i + 1)
                table_string    = table_string + r"\\" + "\n\t" + r"\hline" + "\n"

            # Generate a string for this run
            row_of_table = "\t" + r"\textsc{" + line_name + r"}"
            for std_val in y_axis_std:
                additional_cell = " & " + str(round(std_val, 3))
                row_of_table    = row_of_table + additional_cell
            row_of_table = row_of_table + r"\\" + "\n\t" + r"\hline" + "\n"
            table_string = table_string + row_of_table

        table_string = table_string + r"\end{tabular}"
        return table_string


    def produce_plot(self):

        # Print each table
        for key in self.plotter_config:
            if key.startswith("table_"):
                table = self.generate_table(key)
                print(table)
                print("==============================================")



        