import cityflow


from pytsc.common.simulator import BaseSimulator
from pytsc.backends.cityflow.retriever import Retriever


class Simulator(BaseSimulator):
    def __init__(self, parsed_network):
        super(Simulator, self).__init__(parsed_network)
        self.engine = None

    @property
    def is_terminated(self):
        if self.sim_step == self.config.simulator["sim_length"]:
            return True
        else:
            return False

    @property
    def sim_step(self):
        return self.sim_time - self.config.simulator["initial_wait_time"]

    @property
    def sim_time(self):
        return self.engine.get_current_time()

    def retrieve_step_measurements(self):
        self.step_measurements = {
            "lane": self.cityflow_retriever.retrieve_lane_measurements(),
            "sim": self.cityflow_retriever.retrieve_sim_measurements(),
        }

    def start_simulator(self):
        self.config.create_and_save_cityflow_cfg()
        # Load CityFlow configuration and create the engine
        thread_num = self.config.simulator["thread_num"]
        self.engine = cityflow.Engine(
            config_file=self.config.cityflow_cfg_file,
            thread_num=thread_num,
        )
        self.cityflow_retriever = Retriever(self)
        for _ in range(self.config.simulator["initial_wait_time"]):
            self.engine.next_step()
        self.retrieve_step_measurements()

    def simulator_step(self, n_steps):
        if n_steps is None:
            n_steps = self.config.simulator["delta_time"]
        if n_steps:
            for _ in range(n_steps):
                self.engine.next_step()
            self.retrieve_step_measurements()

    def close_simulator(self):
        self.engine.reset()
