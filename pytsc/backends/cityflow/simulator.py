import cityflow


from pytsc.common.simulator import BaseSimulator
from pytsc.backends.cityflow.retriever import Retriever


class Simulator(BaseSimulator):
    """
    CityFlow simulator class.

    Args:
        parsed_network (ParsedNetwork): Parsed network object containing network information.
    """
    def __init__(self, parsed_network):
        super(Simulator, self).__init__(parsed_network)
        self.engine = None

    @property
    def is_terminated(self):
        """
        Check if the simulation is terminated.

        Returns:
            bool: True if the simulation is terminated, False otherwise.
        """
        if self.sim_step == self.config.simulator["sim_length"]:
            return True
        else:
            return False

    @property
    def sim_step(self):
        """
        Get the current simulation step.

        Returns:
            int: Current simulation step.
        """
        return self.sim_time - self.config.simulator["initial_wait_time"]

    @property
    def sim_time(self):
        """
        Get the current simulation time.

        Returns:
            float: Current simulation time.
        """
        return self.engine.get_current_time()

    def retrieve_step_measurements(self):
        """
        Retrieve step measurements from the simulator.

        Returns:
            dict: Dictionary containing step measurements.
        """
        self.step_measurements = {
            "lane": self.cityflow_retriever.retrieve_lane_measurements(),
            "sim": self.cityflow_retriever.retrieve_sim_measurements(),
        }

    def start_simulator(self):
        """
        Start the CityFlow simulator.
        """
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
        """
        Perform a simulation step.
        """
        if n_steps is None:
            n_steps = self.config.simulator["delta_time"]
        if n_steps:
            for _ in range(n_steps):
                self.engine.next_step()
            self.retrieve_step_measurements()

    def close_simulator(self):
        """
        Close the CityFlow simulator.
        """
        self.engine.reset()
