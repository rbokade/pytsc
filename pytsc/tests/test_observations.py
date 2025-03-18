import imageio
import matplotlib.pyplot as plt
import numpy as np

from pytsc.common.utils import EnvLogger
from pytsc.controllers.evaluate import Evaluate


class ObservationEvaluator(Evaluate):

    def __init__(
        self,
        scenario,
        simulator_backend,
        controller,
        add_env_args={},
        add_controller_args={},
        **kwargs,
    ):
        super(ObservationEvaluator, self).__init__(
            scenario,
            simulator_backend,
            controller,
            add_env_args,
            add_controller_args,
            **kwargs,
        )
        self.obs_info = self.network.observation_space.get_observation_info()
        self.mat_size = (self.obs_info["max_n_controlled_lanes"], 10)

    # def run(self, hours, save_stats=False, plot_stats=False, output_folder=None):
    #     EnvLogger.log_info(f"Evaluating {self.controller_name} controller")
    #     output_folder = self._create_output_folder(output_folder)
    #     steps = int(hours * 3600 / self.delta_time)
    #     for step in range(steps):
    #         actions = self._get_actions()
    #         _, done, stats = self.network.step(actions)
    #         self._log_stats(step, stats)
    #         if self.network.simulator.is_terminated:
    #             self._init_network()
    #             self._init_controllers()
    #         if done:
    #             self.network.restart()
    #     if save_stats:
    #         self._save_stats(output_folder=output_folder)
    #     if plot_stats:
    #         self._plot_stats(output_folder=output_folder)

    def run(self, hours, output_folder=None):
        output_folder = self._create_output_folder(output_folder)
        steps = int(hours * 3600 / self.delta_time)
        frames_dict = {ts_id: [] for ts_id in self.network.traffic_signals.keys()}
        n_signals = len(self.network.traffic_signals)
        fig, axes = plt.subplots(1, n_signals, figsize=(5 * n_signals, 5))
        if n_signals == 1:
            axes = [axes]
        for step in range(steps):
            actions = self._get_actions()
            _, done, stats = self.network.step(actions)
            obs = self.network.get_observations()
            for idx, (ts_id, ts) in enumerate(self.network.traffic_signals.items()):
                ts_obs = np.asarray(obs[idx])
                # lane_mats = ts_obs[self.obs_info["lane_mat_idxs"]] - 1
                # pos_mats = ts_obs[self.obs_info["pos_mat_idxs"]] + lane_mats
                # pos_mats = pos_mats.reshape(self.mat_size, self.mat_size)
                pos_mats = ts_obs[:-10].reshape(*self.mat_size)
                axes[idx].clear()
                axes[idx].imshow(pos_mats, cmap="viridis")
                axes[idx].set_title(f"Signal {ts_id} at step {step}")
                axes[idx].axis("off")
                frame = plt_to_image(fig)
                frames_dict[ts_id].append(frame)
                if self.network.simulator.is_terminated:
                    self._init_network()
                    self._init_controllers()
                if done:
                    self.network.restart()
        for ts_id, frames in frames_dict.items():
            gif_path = (
                f"{output_folder}/{self.scenario}_{ts_id}_obs_matrix_animation.gif"
            )
            imageio.mimsave(gif_path, frames, fps=10)
            print(f"Saved GIF for {ts_id} at {gif_path}")

        # def run(self, hours, output_folder=None):
        #     output_folder = self._create_output_folder(output_folder)
        #     steps = int(hours * 3600 / self.delta_time)
        #     frames = []
        #     fig, ax = plt.subplots(figsize=(6, 6))
        #     for step in range(steps):
        #         actions = self._get_actions()
        #         _, done, stats = self.network.step(actions)
        #         pos_mat = self.network.simulator.step_measurements["sim"]["position_matrix"]
        #         global_matrix = pos_mat
        #         global_matrix = global_matrix.reshape(268, 366)
        #         ax.clear()
        #         ax.imshow(global_matrix, cmap="viridis")
        #         ax.set_title(f"Global Observation Matrix at step {step}")
        #         ax.axis("off")
        #         # plt.pause(0.01)
        #         frame = plt_to_image(fig)
        #         frames.append(frame)
        #         if self.network.simulator.is_terminated:
        #             self._init_network()
        #             self._init_controllers()
        #         if done:
        #             self.network.restart()

        gif_path = f"{output_folder}/{self.scenario}_global_obs_matrix_animation.gif"
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"Saved global observation GIF at {gif_path}")


def plt_to_image(fig):
    """
    Converts a matplotlib figure to an RGB image represented as a NumPy array.
    """
    fig.canvas.draw()  # Draw the figure
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


if __name__ == "__main__":

    scenario = "random_grid_singles"
    output_folder = "/home/rohitbokade/repos/pytsc"
    add_env_args = {
        "signal": {"observation_space": "position_matrix", "obs_dropout_prob": 0.5}
    }

    obs_evaluator = ObservationEvaluator(
        scenario=scenario,
        simulator_backend="sumo",
        controller="sotl",
        add_env_args=add_env_args,
    )
    obs_evaluator.run(hours=0.1, output_folder=output_folder)
