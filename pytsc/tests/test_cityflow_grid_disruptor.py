from pytsc.backends.cityflow.grid_disruptor import CityFlowGridDisruptor

if __name__ == "__main__":

    for disruption_ratio in [0.25, 0.5, 0.75, 1.0]:
        for speed_reduction_factor in [1.0, 0.75, 0.5, 0.25]:
            grid_disruptor = CityFlowGridDisruptor(
                scenario="syn_5x5",
                disruption_ratio=disruption_ratio,
                speed_reduction_factor=speed_reduction_factor,
            )
            grid_disruptor.generate_disrupted_network()
