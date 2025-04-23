Usage
=====

Here is a simple example of using PyTSC with a SUMO grid network:

.. code-block:: python

    from pytsc.environment import TrafficSignalNetwork

    env = TrafficSignalNetwork(scenario='2x2_grid', simulator='sumo')
    obs = env.reset()
    done = False

    while not done:
        action = {id: env.action_space.sample() for id in env.agent_ids}
        obs, reward, done, info = env.step(action)
