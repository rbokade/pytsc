import os

import pandas as pd
import matplotlib.pyplot as plt

from pytsc.controllers import (
    GreedyPhaseSelector,
    FixedTimePhaseSelector,
    MaxPressurePhaseSelector,
    SOTLPhaseSelector,
)


class Evaluate:
    def __init__(self, traffic_signal_network, controller, **kwargs):
        self.kwargs = kwargs
        self.controller = controller
        self.network = traffic_signal_network
        self.config = self.network.config
        self.scenario = self.network.scenario
        self.simulator_type = self.network.simulator_type
        self.env_config = getattr(self.config, f"{self.simulator_type}_config")
        self.delta_time = self.env_config["delta_time"]
        self.steps_per_hour = int(3600 / self.delta_time)
        self._init_controllers()
        self.log = {}

    def _init_controllers(self):
        self.controllers = {}
        for ts_id, ts in self.network.traffic_signals.items():
            if self.controller == "greedy":
                controller = GreedyPhaseSelector
            elif self.controller == "fixed_time":
                controller = FixedTimePhaseSelector
            elif self.controller == "max_pressure":
                controller = MaxPressurePhaseSelector
            elif self.controller == "sotl":
                controller = SOTLPhaseSelector
            else:
                raise ValueError(f"Controller {self.controller} not supported.")
            self.controllers[ts_id] = controller(ts, **self.kwargs)

    def run(self, hours, save_stats=False, plot_stats=False, output_folder=None):
        output_folder = self._create_output_folder(output_folder)
        steps = int(hours * self.steps_per_hour)
        for step in range(1, steps + 1):
            actions = self._get_actions()
            _, done, stats = self.network.step(actions)
            self._log_stats(step, stats)
            if done and step < steps:
                self.network.restart()
        if save_stats:
            self._save_stats(output_folder=output_folder)
        if plot_stats:
            self._plot_stats(output_folder=output_folder)

    def _create_output_folder(self, output_folder):
        if output_folder is None:
            output_folder = os.path.join(
                "pytsc", "results", self.simulator_type, self.scenario
            )
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    def _get_actions(self):
        actions = []
        action_masks = self.network.get_action_mask()
        for idx, (ts_id, ts) in enumerate(self.network.traffic_signals.items()):
            inp = self._get_controller_input(ts)
            action_mask = action_masks[idx]
            action = self.controllers[ts_id].get_action(inp, action_mask)
            actions.append(action)
        return actions

    def _get_controller_input(self, ts):
        inp = self.network.simulator.step_measurements
        inp.update(
            {
                "time": self.network.simulator.sim_time,
                "current_phase_index": ts.controller.program.current_phase_index,
                "time_on_phase": ts.controller.time_on_phase,
                "time_on_cycle": ts.controller.time_on_cycle,
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
        file = os.path.join(output_folder, f"{self.controller}_stats.csv")
        stats.to_csv(file, index=False)

    def _plot_stats(self, output_folder):
        file = os.path.join(output_folder, f"{self.controller}_stats.png")
        num_stats = len(self.log.keys())
        ncols = 3
        nrows = (num_stats + ncols - 1) // ncols
        figsize = (4 * ncols, 2.5 * nrows)
        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        if nrows == 1:
            axes = [axes]
        for idx, (name, value) in enumerate(self.log.items()):
            ax = axes[idx // ncols, idx % ncols]
            t = [i * self.delta_time for i in range(len(value))]
            ax.plot(t, value, label=name, alpha=0.75)
            ax.set_title(name.replace("_", " ").capitalize())
            ax.set_xlabel("Time")
            ax.set_ylabel(name.replace("_", " ").capitalize())
            ax.grid(linestyle="--")
        for idx in range(num_stats, nrows * ncols):
            axes[idx // ncols, idx % ncols].axis("off")
        plt.tight_layout()
        plt.savefig(file)
        plt.show()
