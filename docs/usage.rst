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

To use PyTSC with PyMARL or EPyMARL, you can use the `pymarl` or `epymarl` wrappers. 

Here's an example of how to use PyTSC with EPyMARL. In `src/envs/__init__.py`, you can register the PyTSC environment as such:

.. code-block:: python

    def register_pytsc():
        from pytsc.wrappers.epymarl import EPyMARLTrafficSignalNetwork

        def pytsc_fn(**kwargs) -> MultiAgentEnv:
            kwargs = __check_and_prepare_pytsc_kwargs(**kwargs)
            return EPyMARLTrafficSignalNetwork(**kwargs)

        REGISTRY["pytsc"] = pytsc_fn

Then, you can add config as such: [`src/configs/envs/pasubio.yaml`]

.. code-block:: yaml

    env: pytsc
    
    env_args:
    scenario: pasubio
    map_name: pasubio
    simulator_backend: "sumo"
    sumo:
        episode_limit: 360
    misc:
        reward_gamma: 0.0
        return_lane_stats: False
        return_agent_stats: False
    signal: 
        observation_space: position_matrix
        visibility: 10
    network:
        control_scheme: decentralized

    agent: tsc
    test_greedy: True
    test_nepisode: 10  # (10 episodes = 1 simulation hour, 1 episode = 72 pymarl steps)
    test_interval: 14400  # after 20 episodes
    log_interval: 1440  # after 10 episodes
    runner_log_interval: 1440
    learner_log_interval: 1440
    save_model: True
    save_model_interval: 144000  # after 100 episodes  

    t_max: 2880000  

    use_cuda: True
    use_tensorboard: True