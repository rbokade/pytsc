import matplotlib.pyplot as plt
import numpy as np

from pytsc.common.utils import EnvLogger
from pytsc.controllers.evaluate import Evaluate


NETWORK_INFO = {
    "hangzhou_4_4": {
        "simulator_backend": "cityflow",
        "controller": "fixed_time",
        "n_lanes_per_side": 6,
        "intersection_1_1_lane_map": {
            "road_0_1_0_0": {"orientation": "EB", "pos": 10},
            "road_0_1_0_1": {"orientation": "EB", "pos": 11},
            "road_0_1_0_2": {"orientation": "EB", "pos": 12},
            "road_1_0_1_0": {"orientation": "NB", "pos": 10},
            "road_1_0_1_1": {"orientation": "NB", "pos": 11},
            "road_1_0_1_2": {"orientation": "NB", "pos": 12},
            "road_1_2_3_0": {"orientation": "SB", "pos": 10},
            "road_1_2_3_1": {"orientation": "SB", "pos": 11},
            "road_1_2_3_2": {"orientation": "SB", "pos": 12},
            "road_2_1_2_0": {"orientation": "WB", "pos": 10},
            "road_2_1_2_1": {"orientation": "WB", "pos": 11},
            "road_2_1_2_2": {"orientation": "WB", "pos": 12},
        },
    }
}


class ObservationEvaluator(Evaluate):

    def __init__(self, scenario, simulator_backend, controller, **kwargs):
        super(ObservationEvaluator, self).__init__(
            scenario, simulator_backend, controller, **kwargs
        )
        mat_size = (
            2 * self.config.signal["visibility"]
            + NETWORK_INFO[scenario]["n_lanes_per_side"]
        )
        self.obs_matrix = np.zeros((mat_size, mat_size), dtype=np.float32)
        self.network.traffic_signals["intersection_1_1"].debug = True

    def run(
        self, hours, save_stats=False, plot_stats=False, output_folder=None
    ):
        EnvLogger.log_info(f"Evaluating {self.controller_name} controller")
        output_folder = self._create_output_folder(output_folder)
        steps = int(hours * self.steps_per_hour)
        # Initialize the plot
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(10, 10))
        mat = ax.matshow(np.zeros((26, 26)), cmap="Reds", vmin=0, vmax=1)
        plt.colorbar(mat)
        for step in range(1, steps + 1):
            # ============================================== #
            lane_map = NETWORK_INFO[scenario]["intersection_1_1_lane_map"]
            intersection_1_1 = self.network.traffic_signals["intersection_1_1"]
            for lane in intersection_1_1.incoming_lanes:
                pos_mat = intersection_1_1.lane_pos_mats[lane]
                if lane_map[lane]["orientation"] == "EB":
                    self.obs_matrix[lane_map[lane]["pos"], 0:10] = pos_mat
                if lane_map[lane]["orientation"] == "NB":
                    self.obs_matrix[0:10, lane_map[lane]["pos"]] = pos_mat
                if lane_map[lane]["orientation"] == "SB":
                    self.obs_matrix[16:26, lane_map[lane]["pos"]] = pos_mat
                if lane_map[lane]["orientation"] == "WB":
                    self.obs_matrix[lane_map[lane]["pos"], 16:26] = pos_mat
            mat.set_data(self.obs_matrix)
            # Redraw the figure
            fig.canvas.draw()
            fig.canvas.flush_events()
            # ============================================== #
            actions = self._get_actions()
            _, done, stats = self.network.step(actions)
            self._log_stats(step, stats)
            if done and step < steps:
                self.network.restart()
        if plot_stats:
            pass
        plt.ioff()
        plt.show()


if __name__ == "__main__":

    scenario = "hangzhou_4_4"
    obs_evaluator = ObservationEvaluator(
        scenario=scenario,
        simulator_backend=NETWORK_INFO[scenario]["simulator_backend"],
        controller=NETWORK_INFO[scenario]["controller"],
    )
    obs_evaluator.run(hours=1)
