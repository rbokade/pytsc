PyTSC is a Python library for **traffic signal control and reinforcement learning**. It lets you simulate traffic scenarios and train agents using different MARL algorithms. You can use tools like **SUMO or CityFlow** as the traffic simulator. The library is **modular**. *You can plug in different backends or RL frameworks.* It works well for research on traffic control and multi-agent systems. 

PyTSC is **open-source**. You can find it on `GitHub <https://github.com/rbokade/pytsc>`__.

Users are welcome to contribute or share their work with others.

Installation
============

To install **PyTSC**, clone the repository and install its dependencies:

.. code-block:: bash

    git clone https://github.com/rbokade/pytsc
    cd pytsc
    pip install -e .

Required Simulators:
--------------------

PyTSC supports multiple simulation backends. You can choose one based on your requirements:

- `SUMO <https://www.eclipse.org/sumo/>`__
- `CityFlow <https://cityflow-project.github.io/>`__

Required MARL Libraries:
------------------------

PyTSC is compatible with several Multi-Agent Reinforcement Learning (MARL) frameworks. You can use any of the following frameworks to train your agents:

- `PyMARL <https://github.com/oxwhirl/pymarl>`__
- `EPyMARL <https://github.com/uoe-agents/epymarl>`__
- `Ray RLlib <https://docs.ray.io/en/latest/rllib/index.html>`__
- `MARLlib <https://github.com/Replicable-MARL/MARLlib>`__

