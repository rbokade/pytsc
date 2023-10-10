import logging

from pytsc.common.actions import BaseActionSpace
from pytsc.common.observations import BaseObservationSpace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficSignalNetwork:
    def __init__(self, scenario, simulator_type="sumo", add_config={}):
        (
            Config,
            Simulator,
            NetworkParser,
        ) = import_simulator_modules(simulator_type)
        self.scenario = scenario
        self.simulator_type = simulator_type
        self.config = Config(scenario, add_config)
        self.parsed_network = NetworkParser(self.config)
        self.simulator = Simulator(self.parsed_network)
        self.simulator.start_simulator()
        self._get_env_config()
        self._init_traffic_signals()
        self._init_parsers()
        self.hour_count = 0
        self.episode_count = 0

    @property
    def episode_limit(self):
        return int(
            self.env_config["episode_limit"] / self.env_config["delta_time"]
        )

    @property
    def episode_over(self):
        return (
            self.simulator.sim_step
            % (
                self.env_config["episode_limit"]
                / self.env_config["delta_time"]
            )
            == 0
        )

    def _get_env_config(self):
        if self.simulator_type == "sumo":
            self.env_config = self.config.sumo_config
        elif self.simulator_type == "cityflow":
            self.env_config = self.config.cityflow_config
        else:
            raise ValueError(
                "Simulator type {} is not supported".format(
                    self.simulator_type
                )
            )

    def _init_parsers(self):
        MetricsParser = import_metrics_parser_module(self.simulator_type)
        self.observations = BaseObservationSpace(
            self.config,
            self.parsed_network,
            self.traffic_signals,
            self.simulator_type,
        )
        self.metrics = MetricsParser(
            self.parsed_network,
            self.simulator,
            self.traffic_signals,
        )
        self.actions = BaseActionSpace(self.traffic_signals)

    def _init_traffic_signals(self):
        TrafficSignal = import_traffic_signal_module(self.simulator_type)
        parsed_traffic_signals = self.parsed_network.traffic_signals
        self.traffic_signals = {}
        for ts_id, signal_config in parsed_traffic_signals.items():
            self.traffic_signals[ts_id] = TrafficSignal(
                ts_id, signal_config, self.simulator
            )
            self.traffic_signals[ts_id].update_stats(
                self.simulator.step_measurements
            )

    def _update_ts_stats(self):
        for ts_id in self.traffic_signals.keys():
            self.traffic_signals[ts_id].update_stats(
                self.simulator.step_measurements
            )

    def get_action_mask(self):
        return self.actions.get_mask()

    def get_action_size(self):
        return self.actions.get_size()

    def get_observations(self):
        return self.observations.get_observations()

    def get_observation_size(self):
        return self.observations.get_size()

    def get_reward(self):
        return self.metrics.reward

    def get_local_rewards(self):
        return self.metrics.rewards

    def get_local_rewards_size(self):
        n_traffic_signals = len(self.traffic_signals)
        return int(n_traffic_signals)

    def get_outgoing_density_map(self):
        return self.metrics.outgoing_density_map

    def get_env_info(self):
        stats = self.metrics.get_step_stats()
        stats.update({"episode_count": self.episode_count})
        stats.update({"episode_limit": self.episode_limit})
        return stats

    def get_env_stats(self):
        stats = self.get_env_info()
        for v in self.simulator.step_measurements.values():
            stats.update(v)
        return stats

    def restart(self):
        if self.simulator.is_terminated:
            self.simulator.close_simulator()
            self.simulator.start_simulator()
            self._init_traffic_signals()
            self._init_parsers()
            self.hour_count += 1
        self.episode_count += 1

    def step(self, actions):
        self.actions.apply(actions)
        self.simulator.simulator_step(n_steps=self.env_config["delta_time"])
        self._update_ts_stats()
        return self.get_reward(), self.episode_over, self.get_env_info()


def import_simulator_modules(simulator_type):
    if simulator_type == "sumo":
        from pytsc.sumo.config import Config
        from pytsc.sumo.simulator import Simulator
        from pytsc.sumo.network_parser import NetworkParser
    elif simulator_type == "cityflow":
        from pytsc.city_flow.config import Config
        from pytsc.city_flow.simulator import Simulator
        from pytsc.city_flow.network_parser import NetworkParser
    else:
        raise ValueError(
            "Invalid simulator type. Must be either 'sumo' or 'cityflow'."
        )
    return (
        Config,
        Simulator,
        NetworkParser,
    )


def import_traffic_signal_module(simulator_type):
    if simulator_type == "sumo":
        from pytsc.sumo.traffic_signal import TrafficSignal
    elif simulator_type == "cityflow":
        from pytsc.city_flow.traffic_signal import TrafficSignal
    else:
        raise ValueError(
            "Invalid simulator type. Must be either 'sumo' or 'cityflow'."
        )
    return TrafficSignal


def import_metrics_parser_module(simulator_type):
    if simulator_type == "sumo":
        from pytsc.sumo.metrics import MetricsParser
    elif simulator_type == "cityflow":
        from pytsc.city_flow.metrics import MetricsParser
    else:
        raise ValueError(
            "Invalid simulator type. Must be either 'sumo' or 'cityflow'."
        )
    return MetricsParser
