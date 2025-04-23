.. PyTSC documentation master file, created by
   sphinx-quickstart on Wed Apr 23 11:28:52 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/pytsc_logo.png
   :alt: PyTSC Logo
   :align: center
   :scale: 40%

PyTSC is a Python library for training reinforcement learning (RL) models for traffic signal control (TSC). It lets you simulate traffic scenarios and train agents using different RL/MARL algorithms. You can either **SUMO or CityFlow** simulator. The library is **modular** so *you can modify MDP formulations as per as your need.* PyTSC is **open-source**. You can find it on `GitHub <https://github.com/rbokade/pytsc>`__. Contributions from the community are welcome and appreciated!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api

.. note:: Citation

   If you use this repository for your research, please cite it using the following BibTeX entry:

   .. code-block:: bibtex

      @article{bokade2025pytsc,
        title={Pytsc: A unified platform for multi-agent reinforcement learning in traffic signal control},
        author={Bokade, Rohit and Jin, Xiaoning},
        journal={Sensors},
        volume={25},
        number={5},
        pages={1302},
        year={2025},
        publisher={MDPI}
      }