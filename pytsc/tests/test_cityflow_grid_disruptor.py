from pytsc.backends.cityflow.grid_disruptor import CityFlowGridDisruptor

if __name__ == "__main__":

    for scenario in [2, 3, 4, 5]:
        for disruption_ratio in [0.25, 0.5, 0.75, 1.0]:
            for speed_reduction_factor in [1.0, 0.75, 0.5, 0.25]:
                for replicate_no in range(1, 101):
                    grid_disruptor = CityFlowGridDisruptor(
                        scenario=f"syn_{scenario}x{scenario}",
                        disruption_ratio=disruption_ratio,
                        speed_reduction_factor=speed_reduction_factor,
                        replicate_no=replicate_no,
                    )
                    grid_disruptor.generate_disrupted_network()
