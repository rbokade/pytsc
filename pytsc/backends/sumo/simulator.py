import gc
import os
import subprocess
import sys
import time

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
    import traci
    from sumolib import checkBinary
    from sumolib.miscutils import getFreeSocketPort
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from pytsc.backends.sumo.retriever import Retriever
from pytsc.common.simulator import BaseSimulator


class Simulator(BaseSimulator):
    def __init__(self, parsed_network):
        super(Simulator, self).__init__(parsed_network)
        self.port = None
        self.traci = None
        self.traci_retriever = None

    @property
    def is_terminated(self):
        if self.sim_step == self.config.simulator["sim_length"]:
            return True
        else:
            return False

    @property
    def sim_step(self):
        sim_step = (
            self.sim_time
            - self.parsed_network.config.begin_time
            - self.config.simulator["initial_wait_time"]
        )
        return sim_step

    @property
    def sim_time(self):
        return self.traci.simulation.getTime()

    def retrieve_step_measurements(self):
        self.step_measurements = {
            "ts": self.traci_retriever.retrieve_ts_measurements(),
            "lane": self.traci_retriever.retrieve_lane_measurements(),
            "sim": self.traci_retriever.retrieve_sim_measurements(),
        }

    def start_simulator(self):
        self.port = getFreeSocketPort()
        self.config.reset_config()
        if self.config.simulator["render"]:
            sumo_binary = checkBinary("sumo-gui")
        else:
            sumo_binary = checkBinary("sumo")
        cmd = [sumo_binary, "-c", self.config.sumo_cfg_dir]
        cmd += ["--start"]
        cmd += ["--no-warnings", "True"]
        cmd += [
            "--time-to-teleport",
            str(self.config.simulator["time_to_teleport"]),
        ]
        cmd += ["--quit-on-end", "True"]
        cmd += ["--no-step-log", "True"]
        cmd += ["--duration-log.disable", "True"]
        cmd += ["--remote-port", str(self.port)]
        subprocess.Popen(cmd)
        time.sleep(1)
        self.traci = traci.connect(self.port)
        self.traci_retriever = Retriever(self)
        self.traci_retriever.subscribe()
        if self.config.simulator["initial_wait_time"]:
            for _ in range(self.config.simulator["initial_wait_time"]):
                self.traci.simulationStep()
        self.retrieve_step_measurements()

    def simulator_step(self, n_steps):
        if n_steps is None:
            n_steps = self.config.simulator["delta_time"]
        if n_steps:
            for _ in range(n_steps):
                self.traci.simulationStep()
            self.retrieve_step_measurements()

    def close_simulator(self):
        if self.traci is not None:
            self.traci.close()
            gc.collect()
        self.port = None
        self.traci = None
        self.traci_retriever = None
