import os
import matplotlib.pyplot as plt
import pandas as pd

from pytsc.traffic_signal_network import TrafficSignalNetwork
from pytsc.controllers.evaluate import Evaluate


def run_evaluation(scenario, simulator_type, controller, hours=1, add_args={}):
    traffic_signal_network = TrafficSignalNetwork(scenario, simulator_type)
    evaluate = Evaluate(traffic_signal_network, controller, **add_args)
    evaluate.run(hours, save_stats=True, plot_stats=True)
    stats = pd.DataFrame(evaluate.log)
    stats["controller"] = controller
    return stats


def save_stats_to_file(
    all_stats, simulator_type, scenario, output_folder=None
):
    fname = os.path.join(output_folder, "stats.csv")
    all_stats.to_csv(fname, index=False)


def plot_stats(all_stats, controllers, scenario, output_folder):
    stat_names = [stat for stat in all_stats.columns if stat != "controller"]
    num_stats = len(stat_names)
    ncols = 3
    nrows = (num_stats + ncols - 1) // ncols
    figsize = (4 * ncols, 2.5 * nrows)
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1:
        axes = [axes]
    for idx, stat_name in enumerate(stat_names):
        ax = axes[idx // ncols, idx % ncols]
        for controller in controllers:
            controller_stats = all_stats.loc[
                all_stats.controller == controller, stat_name
            ]
            t = [i * 5 for i in range(len(controller_stats))]
            ax.plot(t, controller_stats, label=controller, alpha=0.5)
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel(stat_name.replace("_", " ").capitalize())
        ax.grid(linestyle="--")
    for idx in range(num_stats, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")
    fname = os.path.join(output_folder, f"{scenario}_stats.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()


def evaluate_controllers(
    scenario,
    simulator_type,
    controllers,
    hours=1,
    output_folder=None,
    add_args={},
):
    if output_folder is None:
        output_folder = "pytsc/results"
    output_folder = os.path.join(output_folder, simulator_type, scenario)
    os.makedirs(output_folder, exist_ok=True)
    all_stats = []
    for controller in controllers:
        stats = run_evaluation(
            scenario,
            simulator_type,
            controller,
            hours=hours,
            add_args=add_args.get(controller, {}),
        )
        all_stats.append(stats)
    all_stats = pd.concat(all_stats, axis=0, ignore_index=True)
    save_stats_to_file(all_stats, simulator_type, scenario, output_folder)
    plot_stats(all_stats, controllers, scenario, output_folder)


if __name__ == "__main__":

    scenario = "ingolstadt7"
    simulator_type = "sumo"
    controllers = ["fixed_time", "greedy", "max_pressure", "sotl"]
    # controllers = ["sotl"]
    hours = 1
    add_args = {
        "fixed_time": {"green_time": 25},
        "sotl": {"phi": 1, "mu": 3, "omega": 30},
    }
    evaluate_controllers(
        scenario,
        simulator_type,
        controllers,
        output_folder="pytsc/results",
        hours=hours,
        add_args=add_args,
    )
