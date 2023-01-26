from streamline import plots

import argparse
import json
import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_loc", default=None, type=str, help="The location where the plot will be saved")
    parser.add_argument("plotter_config", type=str, help="Location of config file (json)")
    args = parser.parse_args()

    json_config_loc = args.plotter_config
    with open(json_config_loc, "r") as f:
        plotter_config = json.load(f)

    if args.save_loc is None:
        plot_save_location = F"{plotter_config['name']}.png"
    else:
        plot_save_location = args.save_loc

    if plotter_config["type"] == "one_plot":
        plotter = plots.OnePlotter(plotter_config)
    elif plotter_config["type"] == "row_plot":
        plotter = plots.RowPlotter(plotter_config)
    elif plotter_config["type"] == "bar_plot":
        plotter = plots.BarPlotter(plotter_config)
    elif plotter_config["type"] == "std_table":
        plotter = plots.STDTable(plotter_config)
    else:
        raise ValueError("Invalid plot type")
    
    plotter.produce_plot()
    if type(plotter) != plots.STDTable:
        plt.savefig(plot_save_location)