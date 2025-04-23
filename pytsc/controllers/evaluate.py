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
    """
    Base class for evaluating traffic signal controllers.
    Args:
        scenario (str): Name of the scenario to evaluate.
        simulator_backend (str): Simulator backend to use (e.g., "cityflow", "sumo").
        controller_name (str): Name of the controller to evaluate.
        add_env_args (dict): Additional arguments for the environment.
        add_controller_args (dict): Additional arguments for the controller.
        **kwargs: Additional keyword arguments.
    """
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
        """
        Initialize controllers for each traffic signal in the network.
        """
        self.controllers = {}
        for ts_id, ts in self.network.traffic_signals.items():
            self.controllers[ts_id] = CONTROLLERS[self.controller_name](
                ts, **self.add_controller_args
            )

    def run(self, hours, save_stats=False, plot_stats=False, output_folder=None):
        """
        Run the evaluation for a specified number of hours.
        Args:
            hours (int): Number of hours to evaluate.
            save_stats (bool): Flag to save statistics.
            plot_stats (bool): Flag to plot statistics.
            output_folder (str): Folder to save output files.
        """
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
        """
        Create the output folder for saving results.
        Args:
            output_folder (str): Folder to save output files.
        Returns:    
            str: Path to the output folder.
        """
        if output_folder is None:
            output_folder = os.path.join(
                "pytsc", "results", self.simulator_backend, self.scenario
            )
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    def _get_actions(self):
        """
        Get actions for each traffic signal in the network.
        Returns:
            list: List of actions for each traffic signal.
        """
        actions = []
        for idx, (ts_id, ts) in enumerate(self.network.traffic_signals.items()):
            obs = self.network.get_observations()
            inp = self._get_controller_input(obs[idx], ts)
            action = self.controllers[ts_id].get_action(inp)
            actions.append(action)
        return actions

    def _get_controller_input(self, obs, ts):
        """
        Get the input for the controller.
        Args:
            obs (list): Observation for the traffic signal.
            ts (TrafficSignal): Traffic signal object.
        """
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
        """
        Log statistics for the current step.
        Args:
            t (int): Current step.
            stats (dict): Dictionary of statistics to log.
        """
        for stat_name, stat_value in stats.items():
            if stat_name not in self.log:
                self.log[stat_name] = []
            self.log[stat_name].append(stat_value)

    def _save_stats(self, output_folder):
        """
        Save statistics to a CSV file.
        Args:
            output_folder (str): Folder to save output files.
        """
        stats = pd.DataFrame(self.log)
        file = os.path.join(output_folder, f"{self.controller_name}_stats.csv")
        stats.to_csv(file, index=False)

    def _plot_stats(self, output_folder):
        """
        Plot statistics and save the figure.
        Args:
            output_folder (str): Folder to save output files.
        """
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
    """
    Class for evaluating traffic signal controllers using reinforcement learning.
    Args:
        scenario (str): Name of the scenario to evaluate.
        simulator_backend (str): Simulator backend to use (e.g., "cityflow", "sumo").
        controller_name (str): Name of the controller to evaluate.
        add_env_args (dict): Additional arguments for the environment.
        add_controller_args (dict): Additional arguments for the controller.
        **kwargs: Additional keyword arguments.
    """
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
        """
        Initialize the controller for the network.
        """
        self.controller = CONTROLLERS[self.controller_name](
            self.network, **self.add_controller_args
        )

    def run(self, hours, save_stats=False, plot_stats=False, output_folder=None):
        """
        Run the evaluation for a specified number of hours.
        Args:
            hours (int): Number of hours to evaluate.
            save_stats (bool): Flag to save statistics.
            plot_stats (bool): Flag to plot statistics.
            output_folder (str): Folder to save output files.
        """
        EnvLogger.log_info(f"Evaluating {self.controller_name} controller")
        output_folder = self._create_output_folder(output_folder)
        steps = int(hours * 3600 / self.delta_time)
        hidden_states = self.controller.init_hidden()
        for step in range(steps):
            actions, hidden_states = self._get_actions(hidden_states)
            _, done, stats = self.network.step(actions)
            self._log_stats(step, stats)
            if self.network.simulator.is_terminated:
                self._init_network()
                self._init_controllers()
            if done:
                self.network.restart()
                self.controller.init_hidden()
        if save_stats:
            self._save_stats(output_folder=output_folder)
        if plot_stats:
            self._plot_stats(output_folder=output_folder)

    def _get_actions(self, hidden_states):
        """
        Get actions for each traffic signal in the network.
        Args:
            hidden_states (list): Hidden states for the controller.
        Returns:
            list: List of actions for each traffic signal.
            list: Next hidden states for the controller.
        """
        actions, next_hidden_states = self.controller.get_action(hidden_states)
        return actions.tolist(), next_hidden_states
