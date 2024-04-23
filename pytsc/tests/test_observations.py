import imageio
import matplotlib.pyplot as plt
import numpy as np

from pytsc.common.utils import compute_linearly_weighted_average, EnvLogger
from pytsc.controllers.evaluate import Evaluate


NETWORK_INFO = {
    "jinan_3_4": {
        "simulator_backend": "cityflow",
        "controller": "greedy",
        "n_lanes_per_side": 3,
        "ts_id": "intersection_1_1",
        "lane_map": {
            "road_0_1_0_0": {"orientation": "EB", "pos": 0},
            "road_0_1_0_1": {"orientation": "EB", "pos": 1},
            "road_0_1_0_2": {"orientation": "EB", "pos": 2},
            "road_1_0_1_0": {"orientation": "NB", "pos": 0},
            "road_1_0_1_1": {"orientation": "NB", "pos": 1},
            "road_1_0_1_2": {"orientation": "NB", "pos": 2},
            "road_1_2_3_0": {"orientation": "SB", "pos": 0},
            "road_1_2_3_1": {"orientation": "SB", "pos": 1},
            "road_1_2_3_2": {"orientation": "SB", "pos": 2},
            "road_2_1_2_0": {"orientation": "WB", "pos": 0},
            "road_2_1_2_1": {"orientation": "WB", "pos": 1},
            "road_2_1_2_2": {"orientation": "WB", "pos": 2},
        },
    },
    "2x2_sumo_grid": {
        "simulator_backend": "sumo",
        "controller": "greedy",
        "n_lanes_per_side": 3,
        "ts_id": "A0",
        "lane_map": {
            "A1A0_0": {"orientation": "SB", "pos": 0},
            "A1A0_1": {"orientation": "SB", "pos": 1},
            "A1A0_2": {"orientation": "SB", "pos": 2},
            "B0A0_0": {"orientation": "WB", "pos": 0},
            "B0A0_1": {"orientation": "WB", "pos": 1},
            "B0A0_2": {"orientation": "WB", "pos": 2},
            "bottom0A0_0": {"orientation": "NB", "pos": 0},
            "bottom0A0_1": {"orientation": "NB", "pos": 1},
            "bottom0A0_2": {"orientation": "NB", "pos": 2},
            "left0A0_0": {"orientation": "EB", "pos": 0},
            "left0A0_1": {"orientation": "EB", "pos": 1},
            "left0A0_2": {"orientation": "EB", "pos": 2},
        },
    },
}


class ObservationEvaluator(Evaluate):

    def __init__(self, scenario, simulator_backend, controller, **kwargs):
        super(ObservationEvaluator, self).__init__(
            scenario, simulator_backend, controller, **kwargs
        )
        self.mat_size = (
            2 * self.config.signal["visibility"]
            + NETWORK_INFO[scenario]["n_lanes_per_side"]
        ) - 1
        self.obs_matrix = np.zeros(
            (self.mat_size, self.mat_size), dtype=np.float32
        )
        self.network.traffic_signals[NETWORK_INFO[scenario]["ts_id"]].debug = (
            True
        )

    def run(
        self, hours, save_stats=False, plot_stats=False, output_folder=None
    ):
        EnvLogger.log_info(f"Evaluating {self.controller_name} controller")
        output_folder = self._create_output_folder(output_folder)
        steps = int(hours * self.steps_per_hour)
        # Initialize the plot
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(10, 10))
        mat = ax.matshow(
            np.zeros((self.mat_size, self.mat_size)),
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(mat)
        # Enable the grid
        ax.set_xticks(np.arange(-0.5, self.mat_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.mat_size, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle=":", linewidth=1)
        # Disable major grid lines
        ax.grid(which="major", color="w", linestyle="", linewidth=0)
        frames = []  # To store each frame for the GIF
        visibility = self.config.signal["visibility"]
        for step in range(1, steps + 1):
            # ============================================== #
            lane_map = NETWORK_INFO[scenario]["lane_map"]
            traffic_signal = self.network.traffic_signals[
                NETWORK_INFO[scenario]["ts_id"]
            ]
            pos_mats = compute_linearly_weighted_average(
                traffic_signal.position_matrices
            )
            pos_mats = np.array_split(
                pos_mats, len(traffic_signal.incoming_lanes)
            )
            for idx, lane in enumerate(traffic_signal.incoming_lanes):
                # pos_mat = traffic_signal.lane_pos_mats[lane]
                pos_mat = pos_mats[idx]
                pos = lane_map[lane]["pos"] + visibility
                if lane_map[lane]["orientation"] == "EB":
                    self.obs_matrix[pos, :visibility] = pos_mat
                if lane_map[lane]["orientation"] == "NB":
                    self.obs_matrix[-visibility:, pos] = np.flip(
                        pos_mat, axis=0
                    )
                if lane_map[lane]["orientation"] == "SB":
                    self.obs_matrix[:visibility, pos] = pos_mat
                if lane_map[lane]["orientation"] == "WB":
                    self.obs_matrix[pos, -visibility:] = np.flip(
                        pos_mat, axis=0
                    )
            mat.set_data(self.obs_matrix)
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            # print(step)
            # Save the current figure as an image (in memory)
            img = plt_to_image(fig)
            frames.append(img)
            # # Redraw the figure
            # fig.canvas.draw()
            # fig.canvas.flush_events()
            # ============================================== #
            actions = self._get_actions()
            _, done, stats = self.network.step(actions)
            self._log_stats(step, stats)
            if done and step < steps:
                self.network.restart()
        if plot_stats:
            pass
        # Save frames as a GIF
        imageio.mimsave(
            f"{output_folder}/{self.scenario}_obs_matrix_animation.gif",
            frames,
            fps=10,
        )
        # plt.ioff()
        # plt.show()


def plt_to_image(fig):
    """
    Converts a matplotlib figure to an RGB image represented as a NumPy array.
    """
    fig.canvas.draw()  # Draw the figure
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


if __name__ == "__main__":

    scenario = "2x2_sumo_grid"
    output_folder = "/home/rohitbokade/Desktop/observations/"
    obs_evaluator = ObservationEvaluator(
        scenario=scenario,
        simulator_backend=NETWORK_INFO[scenario]["simulator_backend"],
        controller=NETWORK_INFO[scenario]["controller"],
    )
    obs_evaluator.run(hours=0.1, output_folder=output_folder)
