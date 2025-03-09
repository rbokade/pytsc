import argparse
import cProfile
import io
import multiprocessing
import os
import pstats
from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd

from pytsc.controllers.evaluate import Evaluate, RLEvaluate


def run_evaluation(
    scenario,
    simulator_backend,
    controller,
    hours=1,
    add_env_args={},
    add_controller_args={},
    profile=False,
    output_folder=None,
):

    eval_class = Evaluate if controller in non_rl_controllers else RLEvaluate
    evaluate = eval_class(
        scenario,
        simulator_backend,
        controller,
        add_env_args,
        add_controller_args,
    )

    if profile:
        # Initialize the profiler
        profiler = cProfile.Profile()
        profiler.enable()

    evaluate.run(hours, save_stats=True, plot_stats=False, output_folder=output_folder)

    if profile:
        profiler.disable()
        s = io.StringIO()

        try:
            ps = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.TIME)
        except AttributeError:
            ps = pstats.Stats(profiler, stream=s).sort_stats("time")

        ps.print_stats()
        with open(f"profile_{controller}.txt", "w") as f:
            f.write(s.getvalue())
        print(f"Profiling results saved for {controller} to profile_{controller}.txt")

    stats = pd.DataFrame(evaluate.log)
    stats["controller"] = controller
    return stats


def save_stats_to_file(all_stats, simulator_backend, scenario, output_folder):
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
        ax = axes[idx // ncols][idx % ncols]
        for controller in controllers:
            controller_stats = all_stats.loc[
                all_stats.controller == controller, stat_name
            ]
            t = [i * 5 for i in range(len(controller_stats))]
            ax.plot(t, controller_stats, label=controller, alpha=0.5)
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel(stat_name.replace("_", " ").capitalize())
        ax.grid(linestyle=":")
    for idx in range(num_stats, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fname = os.path.join(output_folder, f"{scenario}_stats.png")
    # plt.tight_layout()
    plt.savefig(fname)


def evaluate_controllers(
    scenario,
    simulator_backend,
    controllers,
    hours=1,
    output_folder=None,
    add_env_args={},
    add_controller_args={},
    profile=False,
    plot=False,
):
    if output_folder is None:
        output_folder = "pytsc/results"
    output_folder = os.path.join(
        "pytsc", "results", simulator_backend, scenario, output_folder
    )
    os.makedirs(output_folder, exist_ok=True)
    all_stats = []
    for controller in controllers:
        stats = run_evaluation(
            scenario,
            simulator_backend,
            controller,
            hours=hours,
            add_env_args=add_env_args,
            add_controller_args=add_controller_args.get(controller, {}),
            profile=profile,
            output_folder=output_folder,
        )
        all_stats.append(stats)
    all_stats = pd.concat(all_stats, axis=0, ignore_index=True)
    save_stats_to_file(all_stats, simulator_backend, scenario, output_folder)
    if plot:
        plot_stats(all_stats, controllers, scenario, output_folder)


def process_flow_files(
    scenario,
    simulator_backend,
    controllers,
    add_env_args,
    add_controller_args,
    hours,
    profile,
    flow_type,
    disruption_ratio,
    idx,
):
    evaluate_controllers(
        scenario,
        simulator_backend,
        controllers,
        output_folder=f"{flow_type}/{disruption_ratio}/{idx}",
        hours=hours,
        add_env_args=add_env_args,
        add_controller_args=add_controller_args,
        profile=profile,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        help="Name of the scenario",
    )
    parser.add_argument(
        "--simulator-backend",
        dest="simulator_backend",
        type=str,
        help="`cityflow` or `sumo`",
    )
    parser.add_argument(
        "--controllers",
        default="all",
        type=str,
        help="`fixed_time`, `sotl`, `max_pressure`, `greedy`, `all`",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling of Evaluate class methods",
    )
    args = parser.parse_args()

    if args.controllers == "all":
        controllers = [
            # # "mixed_rl",
            # "specialized_marl",
            # "multi_generalized_agent",
            "multi_generalized_graph_agent",
            # "single_generalized_agent",
            # "sotl",
            # "greedy",
            # "fixed_time",
            # "max_pressure",
        ]
    else:
        controllers = [args.controllers]

    rl_controllers = [
        "specialized_marl",
        "multi_generalized_agent",
        "multi_generalized_graph_agent",
        "single_generalized_agent",
    ]
    non_rl_controllers = [
        "sotl",
        "greedy",
        "fixed_time",
        "max_pressure",
    ]

    hours = 1
    add_controller_args = {
        "fixed_time": {"green_time": 25},
        "sotl": {"mu": 7, "theta": 5, "phi_min": 5},
    }
    add_env_args = {
        # "disrupted": False,
        "misc": {
            "return_agent_stats": True,
            "return_lane_stats": True,
        },
        # "cityflow": {
        #     "save_replay": True,
        #     "flow_rate_type": "constant",
        #         "flow_file": f"train/flow_disrupted/0_1/0__{args.scenario}__fd_0.1__gaussian_700_flows.json",
        # },
        # "mode": "test",
        # "domain": "flow_disrupted",
        "sumo": {
            "render": True,
            # "sumo_config_file": "random_grid_increased_demand.sumocfg",
        },
    }
    evaluate_controllers(
        args.scenario,
        args.simulator_backend,
        controllers,
        output_folder="baseline",
        hours=hours,
        add_env_args=add_env_args,
        add_controller_args=add_controller_args,
        profile=args.profile,
    )

    sumo_configs = [
        "random_grid_0.1_increased_demand.sumocfg",
        "random_grid_0.2_increased_demand.sumocfg",
        "random_grid_0.3_increased_demand.sumocfg",
        "random_grid_0.4_increased_demand.sumocfg",
        "random_grid_0.5_increased_demand.sumocfg",
    ]
    for sumo_config in sumo_configs:
        demand_increase = sumo_config.split("_")[2]
        add_env_args["sumo"]["sumo_config_file"] = sumo_config
        evaluate_controllers(
            args.scenario,
            args.simulator_backend,
            controllers,
            output_folder=f"demand_increase_{demand_increase}",
            hours=hours,
            add_env_args=add_env_args,
            add_controller_args=add_controller_args,
            profile=args.profile,
        )

    dropouts = [0.1, 0.25, 0.5, 0.75]
    for i, dropout in enumerate(dropouts):
        add_env_args["signal"] = {"obs_dropout_prob": dropout}
        evaluate_controllers(
            args.scenario,
            args.simulator_backend,
            controllers,
            output_folder=f"obs_dropout_{dropout}",
            hours=hours,
            add_env_args=add_env_args,
            add_controller_args=add_controller_args,
            profile=args.profile,
        )

    # flow_files_batch = {
    #     "flow_disrupted": {
    #         0.1: [
    #             f"train/flow_disrupted/0_1/0__{args.scenario}__fd_0.1__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_1/1__{args.scenario}__fd_0.1__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_1/2__{args.scenario}__fd_0.1__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_1/3__{args.scenario}__fd_0.1__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_1/4__{args.scenario}__fd_0.1__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_1/5__{args.scenario}__fd_0.1__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_1/6__{args.scenario}__fd_0.1__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_1/7__{args.scenario}__fd_0.1__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_1/8__{args.scenario}__fd_0.1__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_1/9__{args.scenario}__fd_0.1__gaussian_700_flows.json",
    #         ],
    #         0.2: [
    #             f"train/flow_disrupted/0_2/0__{args.scenario}__fd_0.2__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_2/1__{args.scenario}__fd_0.2__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_2/2__{args.scenario}__fd_0.2__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_2/3__{args.scenario}__fd_0.2__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_2/4__{args.scenario}__fd_0.2__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_2/5__{args.scenario}__fd_0.2__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_2/6__{args.scenario}__fd_0.2__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_2/7__{args.scenario}__fd_0.2__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_2/8__{args.scenario}__fd_0.2__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_2/9__{args.scenario}__fd_0.2__gaussian_700_flows.json",
    #         ],
    #         0.3: [
    #             f"test/flow_disrupted/0_3/0__{args.scenario}__fd_0.3__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_3/1__{args.scenario}__fd_0.3__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_3/2__{args.scenario}__fd_0.3__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_3/3__{args.scenario}__fd_0.3__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_3/4__{args.scenario}__fd_0.3__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_3/5__{args.scenario}__fd_0.3__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_3/6__{args.scenario}__fd_0.3__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_3/7__{args.scenario}__fd_0.3__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_3/8__{args.scenario}__fd_0.3__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_3/9__{args.scenario}__fd_0.3__gaussian_700_flows.json",
    #         ],
    #         0.4: [
    #             f"test/flow_disrupted/0_4/0__{args.scenario}__fd_0.4__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_4/1__{args.scenario}__fd_0.4__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_4/2__{args.scenario}__fd_0.4__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_4/3__{args.scenario}__fd_0.4__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_4/4__{args.scenario}__fd_0.4__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_4/5__{args.scenario}__fd_0.4__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_4/6__{args.scenario}__fd_0.4__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_4/7__{args.scenario}__fd_0.4__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_4/8__{args.scenario}__fd_0.4__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_4/9__{args.scenario}__fd_0.4__gaussian_700_flows.json",
    #         ],
    #         0.5: [
    #             f"train/flow_disrupted/0_5/0__{args.scenario}__fd_0.5__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_5/1__{args.scenario}__fd_0.5__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_5/2__{args.scenario}__fd_0.5__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_5/3__{args.scenario}__fd_0.5__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_5/4__{args.scenario}__fd_0.5__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_5/5__{args.scenario}__fd_0.5__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_5/6__{args.scenario}__fd_0.5__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_5/7__{args.scenario}__fd_0.5__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_5/8__{args.scenario}__fd_0.5__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_5/9__{args.scenario}__fd_0.5__gaussian_700_flows.json",
    #         ],
    #         0.6: [
    #             f"train/flow_disrupted/0_6/0__{args.scenario}__fd_0.6__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_6/1__{args.scenario}__fd_0.6__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_6/2__{args.scenario}__fd_0.6__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_6/3__{args.scenario}__fd_0.6__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_6/4__{args.scenario}__fd_0.6__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_6/5__{args.scenario}__fd_0.6__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_6/6__{args.scenario}__fd_0.6__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_6/7__{args.scenario}__fd_0.6__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_6/8__{args.scenario}__fd_0.6__gaussian_700_flows.json",
    #             f"train/flow_disrupted/0_6/9__{args.scenario}__fd_0.6__gaussian_700_flows.json",
    #         ],
    #         0.7: [
    #             f"test/flow_disrupted/0_7/0__{args.scenario}__fd_0.7__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_7/1__{args.scenario}__fd_0.7__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_7/2__{args.scenario}__fd_0.7__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_7/3__{args.scenario}__fd_0.7__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_7/4__{args.scenario}__fd_0.7__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_7/5__{args.scenario}__fd_0.7__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_7/6__{args.scenario}__fd_0.7__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_7/7__{args.scenario}__fd_0.7__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_7/8__{args.scenario}__fd_0.7__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_7/9__{args.scenario}__fd_0.7__gaussian_700_flows.json",
    #         ],
    #         0.8: [
    #             f"test/flow_disrupted/0_8/0__{args.scenario}__fd_0.8__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_8/1__{args.scenario}__fd_0.8__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_8/2__{args.scenario}__fd_0.8__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_8/3__{args.scenario}__fd_0.8__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_8/4__{args.scenario}__fd_0.8__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_8/5__{args.scenario}__fd_0.8__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_8/6__{args.scenario}__fd_0.8__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_8/7__{args.scenario}__fd_0.8__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_8/8__{args.scenario}__fd_0.8__gaussian_700_flows.json",
    #             f"test/flow_disrupted/0_8/9__{args.scenario}__fd_0.8__gaussian_700_flows.json",
    #         ],
    #     },
    #     "link_disrupted": {
    #         0.1: [
    #             f"train/link_disrupted/0_1/0__{args.scenario}__dr_0.1__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_1/1__{args.scenario}__dr_0.1__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_1/2__{args.scenario}__dr_0.1__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_1/3__{args.scenario}__dr_0.1__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_1/4__{args.scenario}__dr_0.1__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_1/5__{args.scenario}__dr_0.1__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_1/6__{args.scenario}__dr_0.1__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_1/7__{args.scenario}__dr_0.1__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_1/8__{args.scenario}__dr_0.1__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_1/9__{args.scenario}__dr_0.1__gaussian_700_flows.json",
    #         ],
    #         0.2: [
    #             f"test/link_disrupted/0_2/0__{args.scenario}__dr_0.2__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_2/1__{args.scenario}__dr_0.2__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_2/2__{args.scenario}__dr_0.2__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_2/3__{args.scenario}__dr_0.2__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_2/4__{args.scenario}__dr_0.2__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_2/5__{args.scenario}__dr_0.2__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_2/6__{args.scenario}__dr_0.2__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_2/7__{args.scenario}__dr_0.2__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_2/8__{args.scenario}__dr_0.2__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_2/9__{args.scenario}__dr_0.2__gaussian_700_flows.json",
    #         ],
    #         0.3: [
    #             f"train/link_disrupted/0_3/0__{args.scenario}__dr_0.3__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_3/1__{args.scenario}__dr_0.3__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_3/2__{args.scenario}__dr_0.3__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_3/3__{args.scenario}__dr_0.3__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_3/4__{args.scenario}__dr_0.3__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_3/5__{args.scenario}__dr_0.3__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_3/6__{args.scenario}__dr_0.3__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_3/7__{args.scenario}__dr_0.3__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_3/8__{args.scenario}__dr_0.3__gaussian_700_flows.json",
    #             f"train/link_disrupted/0_3/9__{args.scenario}__dr_0.3__gaussian_700_flows.json",
    #         ],
    #         0.4: [
    #             f"test/link_disrupted/0_4/0__{args.scenario}__dr_0.4__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_4/1__{args.scenario}__dr_0.4__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_4/2__{args.scenario}__dr_0.4__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_4/3__{args.scenario}__dr_0.4__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_4/4__{args.scenario}__dr_0.4__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_4/5__{args.scenario}__dr_0.4__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_4/6__{args.scenario}__dr_0.4__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_4/7__{args.scenario}__dr_0.4__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_4/8__{args.scenario}__dr_0.4__gaussian_700_flows.json",
    #             f"test/link_disrupted/0_4/9__{args.scenario}__dr_0.4__gaussian_700_flows.json",
    #         ],
    #     },
    #     "baseline": {
    #         0: [
    #             f"train/flow_disrupted/700/0__{args.scenario}__gaussian_700_flows.json",
    #             f"train/flow_disrupted/700/1__{args.scenario}__gaussian_700_flows.json",
    #             f"train/flow_disrupted/700/2__{args.scenario}__gaussian_700_flows.json",
    #             f"train/flow_disrupted/700/3__{args.scenario}__gaussian_700_flows.json",
    #             f"train/flow_disrupted/700/4__{args.scenario}__gaussian_700_flows.json",
    #             f"train/flow_disrupted/700/5__{args.scenario}__gaussian_700_flows.json",
    #             f"train/flow_disrupted/700/6__{args.scenario}__gaussian_700_flows.json",
    #             f"train/flow_disrupted/700/7__{args.scenario}__gaussian_700_flows.json",
    #             f"train/flow_disrupted/700/8__{args.scenario}__gaussian_700_flows.json",
    #             f"train/flow_disrupted/700/9__{args.scenario}__gaussian_700_flows.json",
    #         ],
    #     },
    # }

    # for flow_type, flow_files in flow_files_batch.items():
    #     for disruption_ratio, flow_files_ in flow_files.items():
    #         add_env_args_list = [
    #             deepcopy(add_env_args) for _ in range(len(flow_files_))
    #         ]
    #         for i, flow_file in enumerate(flow_files_):
    #             add_env_args_list[i]["cityflow"] = {
    #                 "flow_rate_type": "constant",
    #                 "flow_file": flow_file,
    #             }

    #         num_processes = min(len(flow_files_), multiprocessing.cpu_count())
    #         with multiprocessing.Pool(processes=num_processes) as pool:
    #             pool.starmap(
    #                 process_flow_files,
    #                 [
    #                     (
    #                         args.scenario,
    #                         args.simulator_backend,
    #                         controllers,
    #                         add_env_args_,
    #                         add_controller_args,
    #                         hours,
    #                         args.profile,
    #                         flow_type,
    #                         disruption_ratio,
    #                         idx,
    #                     )
    #                     for idx, add_env_args_ in enumerate(add_env_args_list)
    #                 ],
    #             )

    # add_env_args_list = [deepcopy(add_env_args) for _ in range(len(flow_files_batch))]
    # for idx, add_env_args_ in enumerate(add_env_args_list):
    #     add_env_args_["cityflow"] = {
    #         "flow_rate_type": "sequential",
    #         "flow_files": flow_files_batch[idx],
    #     }

    # num_processes = min(len(flow_files_batch), multiprocessing.cpu_count())
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     pool.starmap(
    #         process_flow_files,
    #         [
    #             (
    #                 args.scenario,
    #                 args.simulator_backend,
    #                 controllers,
    #                 add_env_args_,
    #                 add_controller_args,
    #                 hours,
    #                 args.profile,
    #                 idx,
    #             )
    #             for idx, add_env_args_ in enumerate(add_env_args_list)
    #         ],
    #     )
