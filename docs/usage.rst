Usage
=====

Add scenario files
-------------------

Example directory structure:

.. code-block:: text

    pytsc/
    └── scenarios/
        └── sumo/
            └── pasubio/
                ├── pasubio.sumocfg
                ├── pasubio.net.xml
                └── config.yaml

Create a config file
---------------------

Your scenario folder (e.g., `pytsc/scenarios/sumo/pasubio/`) should include a `config.yaml` file like this:

.. code-block:: yaml

    sumo:
      render: False
      sumo_config_file: pasubio.sumocfg
      sumo_net_file: pasubio.net.xml


This config file specifies the simulation parameters. You can find the list of parameters in the `pytsc/scenarios/default/config.yaml` file.

Using PyTSC in your code
-------------------------

Here’s a simple example of using PyTSC with a SUMO grid network:

.. code-block:: python

    from pytsc import TrafficSignalNetwork

    env = TrafficSignalNetwork(scenario='pasubio', simulator='sumo')
    obs = env.reset()
    done = False

    while not done:
        action = {id: env.action_space.sample() for id in env.agent_ids}
        obs, reward, done, info = env.step(action)

Using PyTSC with PyMARL/EPyMARL
-------------------------------

To use PyTSC with PyMARL or EPyMARL, you can use the `pymarl` or `epymarl` wrappers. Here's an example of how to use PyTSC with PyMARL:

.. .. code-block:: python

..     from pytsc import TrafficSignalNetwork
..     from pymarl import Env

..     class PyTSCEnv(Env):
..         def __init__(self, scenario='pasubio', simulator='sumo'):
..             self.env = TrafficSignalNetwork(scenario=scenario, simulator=simulator)

..         def reset(self):
..             return self.env.reset()

..         def step(self, action):
..             return self.env.step(action)

..     env = PyTSCEnv()
..     obs = env.reset()
..     done = False

..     while not done:
..         action = {id: env.action_space.sample() for id in env.agent_ids}
..         obs, reward, done, info = env.step(action)