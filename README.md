![PyTSC Logo](./pytsc_logo.png)

## `PyTSC`: Traffic Signal Control Environment for Multi-Agent Reinforcement Learning (Experimental)

`PyTSC` is a work-in-progress environment aimed at researchers who want to quickly prototype PyTSC-agent reinforcement learning solutions for traffic signal control. The project is designed to offer both [SUMO](https://github.com/eclipse/sumo) and [CityFlow](https://github.com/cityflow-project/CityFlow) simulation backends, allowing researchers to easily integrate open-source traffic signal control datasets for their research projects.

Currently, `PyTSC` is compatible with the [PyMARL](https://github.com/oxwhirl/pymarl) library, providing a smooth experience for researchers already working with it. In future updates, we plan to develop wrappers for other popular MARL libraries to further extend the environment's versatility.

The code is highly modular, enabling researchers to create their own variants of simulations with ease. For SUMO, the environment employs [subscriptions](https://sumo.dlr.de/docs/FAQ.html#traci) to improve simulation speeds. While the integration of the CityFlow simulator is still under development, it remains a priority for the project.

Contributions from the community are welcome and appreciated!

## Generate grid network
1. Create a folder named `{m}x{n}_regular_grid` in the `scenarios` folder
2. Create a `config.yaml` file in the folder and add the required parameters under the key `grid_generator`
3. Run `python pytsc/generators/generate_grid.py --scenario {m}x{n}_regular_grid` to generate the network with the given parameters
4. Create the `.sumocfg` file in the `scenarios` folder (refer to examples)

## Creating trips
`python $SUMO_HOME/tools/randomTrips.py -n {m}x{n}_regular_grid.net.xml -r {m}x{n}_regular_grid.rou.xml --fringe-factor max --begin 0 --end 3600 --period 5 --validate`


## Citation

If you use this repository for your research, please cite it using the following BibTeX entry:

```bibtex
@article{bokade2024pytsc,
  title={PyTSC: A Unified Platform for Multi-Agent Reinforcement Learning in Traffic Signal Control},
  author={Bokade, Rohit and Jin, Xiaoning},
  journal={arXiv preprint arXiv:2410.18202},
  year={2024}
}
