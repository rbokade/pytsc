import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
from numpy import add

from pytsc import TrafficSignalNetwork
from pytsc.common.utils import EnvLogger, validate_input_against_allowed
from pytsc.controllers import CONTROLLERS

# EnvLogger.set_log_level(logging.WARNING)


class Evaluate:
    action_space = "phase_selection"

    def __init__(
        self,
        scenario,
        simulator_backend,
        controller_name,
        add_env_args,
        add_controller_args,
        **kwargs,
    ):
        validate_input_against_allowed(controller_name, CONTROLLERS)
        self.scenario = scenario
        self.simulator_backend = simulator_backend
        self.controller_name = controller_name
        self.add_env_args = add_env_args
        self.add_controller_args = add_controller_args
        self._init_network()
        self._init_controllers()
        self.delta_time = self.config.simulator["delta_time"]
        self.log = {}

    def _init_network(self):
        """
        NOTE: Action space must be set to phase selection
        """
        self.network = TrafficSignalNetwork(
            self.scenario,
            self.simulator_backend,
            **self.add_env_args,
        )
        self.network.config.signal["action_space"] = self.action_space
        self.network._init_parsers()
        self.config = self.network.config

    def _init_controllers(self):
        self.controllers = {}
        for ts_id, ts in self.network.traffic_signals.items():
            self.controllers[ts_id] = CONTROLLERS[self.controller_name](
                ts, **self.add_controller_args
            )

    def run(self, hours, save_stats=False, plot_stats=False, output_folder=None):
        EnvLogger.log_info(f"Evaluating {self.controller_name} controller")
        output_folder = self._create_output_folder(output_folder)
        steps = int(hours * 3600 / self.delta_time)
        for step in range(steps):
            actions = self._get_actions()
            _, done, stats = self.network.step(actions)
            self._log_stats(step, stats)
            if self.network.simulator.is_terminated:
                self._init_network()
                self._init_controllers()
            if done:
                self.network.restart()
        if save_stats:
            self._save_stats(output_folder=output_folder)
        if plot_stats:
            self._plot_stats(output_folder=output_folder)

    def _create_output_folder(self, output_folder):
        if output_folder is None:
            output_folder = os.path.join(
                "pytsc", "results", self.simulator_backend, self.scenario
            )
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    def _get_actions(self):
        actions = []
        for idx, (ts_id, ts) in enumerate(self.network.traffic_signals.items()):
            obs = self.network.get_observations()
            inp = self._get_controller_input(obs[idx], ts)
            action = self.controllers[ts_id].get_action(inp)
            actions.append(action)
        return actions

    def _get_controller_input(self, obs, ts):
        inp = self.network.simulator.step_measurements
        inp.update(
            {
                "observation": obs,
                "time": self.network.simulator.sim_time,
                "current_phase_index": ts.controller.program.current_phase_index,
                "time_on_phase": ts.controller.time_on_phase,
            }
        )
        return inp

    def _log_stats(self, t, stats):
        for stat_name, stat_value in stats.items():
            if stat_name not in self.log:
                self.log[stat_name] = []
            self.log[stat_name].append(stat_value)

    def _save_stats(self, output_folder):
        stats = pd.DataFrame(self.log)
        file = os.path.join(output_folder, f"{self.controller_name}_stats.csv")
        stats.to_csv(file, index=False)

    def _plot_stats(self, output_folder):
        file = os.path.join(output_folder, f"{self.controller_name}_stats.png")
        num_stats = len(self.log.keys())
        ncols = 3
        nrows = (num_stats + ncols - 1) // ncols
        figsize = (4 * ncols, 2.5 * nrows)
        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        if nrows == 1:
            axes = [axes]
        for idx, (name, value) in enumerate(self.log.items()):
            ax = axes[idx // ncols][idx % ncols]
            t = [i * self.delta_time for i in range(len(value))]
            ax.plot(t, value, label=name, alpha=0.75)
            ax.set_title(name.replace("_", " ").capitalize())
            ax.set_xlabel("Time")
            ax.set_ylabel(name.replace("_", " ").capitalize())
            ax.grid(linestyle=":")
        for idx in range(num_stats, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")
        plt.tight_layout()
        plt.savefig(file)
        plt.show()


class RLEvaluate(Evaluate):
    action_space = "phase_switch"

    def __init__(
        self,
        scenario,
        simulator_backend,
        controller_name,
        add_env_args,
        add_controller_args,
        **kwargs,
    ):
        super(RLEvaluate, self).__init__(
            scenario,
            simulator_backend,
            controller_name,
            add_env_args,
            add_controller_args,
            **kwargs,
        )

    def _init_controllers(self):
        self.controller = CONTROLLERS[self.controller_name](
            self.network, **self.add_controller_args
        )

    def run(self, hours, save_stats=False, plot_stats=False, output_folder=None):
        EnvLogger.log_info(f"Evaluating {self.controller_name} controller")
        output_folder = self._create_output_folder(output_folder)
        steps = int(hours * 3600 / self.delta_time)
        for step in range(steps):
            actions = self._get_actions()
            _, done, stats = self.network.step(actions)
            self._log_stats(step, stats)
            if self.network.simulator.is_terminated:
                self._init_network()
                self._init_controllers()
            if done:
                self.network.restart()
        if save_stats:
            self._save_stats(output_folder=output_folder)
        if plot_stats:
            self._plot_stats(output_folder=output_folder)

    def _get_actions(self):
        actions = self.controller.get_action()
        return actions.tolist()
