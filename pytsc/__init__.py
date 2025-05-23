import numpy as np

from pytsc.backends.cityflow import CITYFLOW_MODULES
from pytsc.backends.sumo import SUMO_MODULES
from pytsc.common import ACTION_SPACES, OBSERVATION_SPACES, REWARD_FUNCTIONS
from pytsc.common.actions import CentralizedActionSpace
from pytsc.common.utils import validate_input_against_allowed

SUPPORTED_SIMULATOR_BACKENDS = ("sumo", "cityflow")
REWARD_METRICS = ("queue", "pressure")  # TODO: Implement `pressure`
SIMULATOR_MODULES = {
    "cityflow": CITYFLOW_MODULES,
    "sumo": SUMO_MODULES,
}


class TrafficSignalNetwork:
    def __init__(self, scenario, simulator_backend, **kwargs):
        self.scenario = scenario
        self.simulator_backend = simulator_backend
        self.disrupted = kwargs.get("disrupted", False)
        self.domain_class = kwargs.get("domain_class", None)
        assert (
            self.simulator_backend in SUPPORTED_SIMULATOR_BACKENDS
        ), f"Simulator backend {self.simulator_backend} not supported."
        if self.disrupted:
            self.config = SIMULATOR_MODULES[simulator_backend]["disrupted_config"](
                scenario, **kwargs
            )
        else:
            self.config = SIMULATOR_MODULES[simulator_backend]["config"](
                scenario, **kwargs
            )
        self._validate_config()
        self.parsed_network = SIMULATOR_MODULES[simulator_backend]["network_parser"](
            self.config
        )
        self.simulator = SIMULATOR_MODULES[simulator_backend]["simulator"](
            self.parsed_network
        )
        self.simulator.start_simulator()
        self._init_traffic_signals()
        self._init_parsers()
        self._set_n_agents()
        self._init_counters()

    def _init_counters(self):
        self.hour_count = 0
        self.episode_count = 0

    @property
    def episode_limit(self):
        return int(
            self.config.simulator["episode_limit"] / self.config.simulator["delta_time"]
        )

    @property
    def episode_over(self):
        if self.simulator.sim_step > 0:
            return self.simulator.sim_step % self.config.simulator["episode_limit"] == 0
        else:
            return False

    def _validate_config(self):
        validate_input_against_allowed(
            self.config.signal["action_space"], ACTION_SPACES.keys()
        )
        validate_input_against_allowed(
            self.config.signal["observation_space"], OBSERVATION_SPACES.keys()
        )

    def _init_parsers(self):
        self.action_space = ACTION_SPACES[self.config.signal["action_space"]](
            self.config, self.traffic_signals
        )
        if self.config.network["control_scheme"] == "centralized":
            self.action_space = CentralizedActionSpace(self.action_space)
        self.observation_space = OBSERVATION_SPACES[
            self.config.signal["observation_space"]
        ](
            self.config,
            self.parsed_network,
            self.traffic_signals,
            self.simulator_backend,
        )
        self.metrics = SIMULATOR_MODULES[self.simulator_backend]["metrics_parser"](
            self.parsed_network,
            self.simulator,
            self.traffic_signals,
        )
        self.reward_function = REWARD_FUNCTIONS[self.config.signal["reward_function"]](
            self.metrics, self.traffic_signals
        )

    def _init_traffic_signals(self):
        parsed_traffic_signals = self.parsed_network.traffic_signals
        self.traffic_signals = {}
        for ts_id, signal_config in parsed_traffic_signals.items():
            self.traffic_signals[ts_id] = SIMULATOR_MODULES[self.simulator_backend][
                "traffic_signal"
            ](ts_id, signal_config, self.simulator)
            self.traffic_signals[ts_id].update_stats(self.simulator.step_measurements)

    def _set_n_agents(self):
        self.n_agents = (
            len(self.traffic_signals)
            if self.config.network["control_scheme"] == "decentralized"
            else 1  # centralized
        )

    def _update_ts_stats(self):
        for ts_id in self.traffic_signals.keys():
            self.traffic_signals[ts_id].update_stats(self.simulator.step_measurements)

    def get_action_mask(self):
        return self.action_space.get_mask()

    def get_action_size(self):
        return self.action_space.get_size()

    def get_observations(self):
        if self.config.network["control_scheme"] == "decentralized":
            return self.observation_space.get_observations()
        else:
            return [np.concatenate(self.observation_space.get_observations()).tolist()]

    def get_observation_size(self):
        if self.config.network["control_scheme"] == "decentralized":
            return self.observation_space.get_size()
        else:
            n_a = self.parsed_network.adjacency_matrix.shape[0]
            return self.observation_space.get_size() * n_a

    def get_state(self):
        return self.observation_space.get_state()

    def get_state_size(self):
        return self.observation_space.get_state_size()

    def get_reward(self):
        return self.reward_function.get_global_reward()

    def get_rewards(self):
        if self.config.network["control_scheme"] == "decentralized":
            return self.reward_function.get_local_reward()
        else:
            return [self.reward_function.get_global_reward()]

    def get_env_info(self):
        stats = self.metrics.get_step_stats()
        stats.update({"episode_count": self.episode_count})
        stats.update({"episode_limit": self.episode_limit})
        if self.disrupted:
            stats.update({"n_domains": len(self.config.domain_classes)})
            stats.update({"domain_class": self.config.current_domain_class})
        return stats

    def get_env_stats(self):
        stats = self.get_env_info()
        for v in self.simulator.step_measurements.values():
            stats.update(v)
        return stats

    def restart(self, reset=True):
        if self.episode_over:
            self.episode_count += 1
            # self.observation_space.reset_dropped_lanes()
        if self.simulator.is_terminated:
            self.hour_count += 1
            self.simulator.close_simulator()
            if reset:
                self.simulator.start_simulator()
                if self.domain_class is not None:
                    self.config.set_domain_class(self.domain_class)
                self._init_traffic_signals()
                self._init_parsers()

    def step(self, actions):
        self.action_space.apply(actions)
        self.simulator.simulator_step(n_steps=self.config.simulator["delta_time"])
        self._update_ts_stats()
        return self.get_reward(), self.episode_over, self.get_env_info()
