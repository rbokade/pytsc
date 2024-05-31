from pytsc.backends.cityflow.trip_generator import CityFlowTripGenerator

if __name__ == "__main__":

    # # Generate hourly flows
    # for size in [2, 3, 4, 5]:
    #     for mean_flow in [500, 525, 550, 575, 600, 625, 650, 675, 700]:
    #         trip_generator = CityFlowTripGenerator(
    #             scenario=f"syn_{size}x{size}",
    #             start_time=0,
    #             end_time=3600,
    #             inter_mu=3600 / mean_flow,
    #             inter_sigma=0.80,
    #             turn_probs=[0.1, 0.3, 0.6],
    #         )
    #         trip_generator.generate_flows(
    #             filepath="/home/rohitbokade/repos/pytsc/pytsc/tests"
    #         )

    # Generate disrupted flows
    for size in [2, 3, 4, 5]:
        for disruption_ratio in [0.25, 0.5, 0.75, 1.0]:
            for speed_reduction_factor in [1.0, 0.75, 0.5, 0.25]:
                for replicate_no in range(1, 101):
                    for mean_flow in [600]:
                        scenario = f"syn_{size}x{size}"
                        trip_generator = CityFlowTripGenerator(
                            scenario=scenario,
                            start_time=0,
                            end_time=3600,
                            inter_mu=3600 / mean_flow,
                            inter_sigma=0.80,
                            turn_probs=[0.1, 0.3, 0.6],
                            disrupted=True,
                            disruption_ratio=disruption_ratio,
                            speed_reduction_factor=speed_reduction_factor,
                            replicate_no=replicate_no,
                        )
                        filepath = "/home/rohitbokade/repos/pytsc/"
                        filepath += "pytsc/scenarios/cityflow/"
                        filepath += f"{scenario}/disrupted/"
                        filepath += f"r_{disruption_ratio}__"
                        filepath += f"p_{speed_reduction_factor}/"
                        print(filepath)
                        trip_generator.generate_flows(filepath=filepath)
