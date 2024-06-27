from pytsc.backends.cityflow.trip_generator import (
    CityFlowTripGenerator,
    IntervalCityFlowTripGenerator,
    VariableDemandTripGenerator,
)

if __name__ == "__main__":

    # Generate hourly flows
    # flow_resolution = 25
    # for size in [5]:
    #     for replicate_no in range(1, 21):
    # syn_1x7_edge_weights = {
    #     "road_0_1_0":
    #     "road_1_0_1"
    #     "road_1_1_1"
    #     "road_1_1_2"
    #     "road_0_1_3"
    #     "road_1_1_0"
    #     "road_1_2_3"
    #     "road_2_1_2"
    #     "road_2_0_1"
    #     "road_2_2_3"
    #     "road_2_1_0"
    #     "road_3_1_2"
    #     "road_3_1_1"
    #     "road_3_0_1"
    #     "road_3_1_0"
    #     "road_4_1_2"
    #     "road_4_1_1"
    #     "road_4_2_3"
    #     "road_4_1_3"
    #     "road_4_0_1"
    #     "road_4_1_3"
    #     "road_4_1_0"
    #     "road_5_1_2"
    #     "road_5_1_3"
    #     "road_5_1_0"
    #     "road_5_2_3"
    #     "road_6_1_2"
    #     "road_6_1_1"
    #     "road_6_0_1"
    #     "road_6_1_0"
    #     "road_7_1_2"
    #     "road_7_1_0"
    #     "road_7_1_3"
    #     "road_7_2_3"
    #     "road_8_1_2"
    # }
    syn_1x7_edge_weights = {
        "road_0_1_0": 900 / 2400,
        "road_1_0_1": 1165 / 2400,
        "road_1_1_0": 700 / 2400,
        "road_1_1_1": 1465 / 2400,
        "road_1_1_2": 2300 / 2400,
        "road_1_1_3": 700 / 2400,
        "road_1_2_3": 600 / 2400,
        "road_2_1_0": 650 / 2400,
        "road_2_1_2": 2400 / 2400,
        "road_2_1_3": 250 / 2400,
        "road_2_2_3": 300 / 2400,
        "road_3_0_1": 250 / 2400,
        "road_3_1_0": 550 / 2400,
        "road_3_1_1": 300 / 2400,
        "road_3_1_2": 2300 / 2400,
        "road_4_0_1": 500 / 2400,
        "road_4_1_0": 500 / 2400,
        "road_4_1_1": 425 / 2400,
        "road_4_1_2": 2250 / 2400,
        "road_4_1_3": 325 / 2400,
        "road_4_2_3": 450 / 2400,
        "road_5_1_0": 500 / 2400,
        "road_5_2_3": 250 / 2400,
        "road_5_1_2": 2000 / 2400,
        "road_5_1_3": 200 / 2400,
        "road_6_1_0": 500 / 2400,
        "road_6_1_1": 200 / 2400,
        "road_6_1_2": 1950 / 2400,
        "road_6_0_1": 250 / 2400,
        "road_7_1_0": 550 / 2400,
        "road_7_2_3": 250 / 2400,
        "road_7_1_2": 1900 / 2400,
        "road_7_1_3": 200 / 2400,
        "road_8_1_2": 1900 / 2400,
    }
    inter_mus = {
        "road_0_1_0": 3600 / 900,
        "road_1_0_1": 3600 / 1165,
        "road_1_2_3": 3600 / 600,
        "road_2_2_3": 3600 / 300,
        "road_3_0_1": 3600 / 250,
        "road_4_0_1": 3600 / 500,
        "road_4_2_3": 3600 / 450,
        "road_5_2_3": 3600 / 250,
        "road_6_0_1": 3600 / 250,
        "road_8_1_2": 3600 / 1900,
        "road_7_2_3": 3600 / 250,
    }
    inter_sigmas = {k: 1.2 for k in inter_mus.keys()}
    for mean_flow in [600]:
        trip_generator = VariableDemandTripGenerator(
            scenario="syn_1x7",
            start_time=0,
            end_time=3600,
            inter_mus=inter_mus,
            inter_sigmas=inter_sigmas,
            turn_probs=[0.33, 0.33, 0.33],
            edge_weights=syn_1x7_edge_weights,
        )
        trip_generator.generate_flows(
            filepath="/Users/rohitbokade/repos/pytsc/pytsc/tests",
        )

    # # Generate hourly flows
    # flow_resolution = 25
    # for size in [4]:
    #     # for size in [2, 3, 4, 5]:
    #     for mean_flow in list(
    #         range(100, 700 + flow_resolution, flow_resolution)
    #     ):
    #         trip_generator = CityFlowTripGenerator(
    #             scenario=f"syn_{size}x{size}",
    #             start_time=0,
    #             end_time=3600,
    #             inter_mu=3600 / mean_flow,
    #             inter_sigma=0.80,
    #             turn_probs=[0.1, 0.3, 0.6],
    #         )
    #         trip_generator.generate_flows(
    #             filepath="/Users/rohitbokade/repos/pytsc/pytsc/tests"
    #         )

    # # Generate disrupted flows
    # for size in [2, 3, 4, 5]:
    #     for disruption_ratio in [0.25, 0.5, 0.75, 1.0]:
    #         for speed_reduction_factor in [0.75, 0.5, 0.25]:
    #             for replicate_no in range(1, 101):
    #                 for mean_flow in [600]:
    #                     scenario = f"syn_{size}x{size}"
    #                     trip_generator = CityFlowTripGenerator(
    #                         scenario=scenario,
    #                         start_time=0,
    #                         end_time=3600,
    #                         inter_mu=3600 / mean_flow,
    #                         inter_sigma=0.80,
    #                         turn_probs=[0.1, 0.3, 0.6],
    #                         disrupted=True,
    #                         disruption_ratio=disruption_ratio,
    #                         speed_reduction_factor=speed_reduction_factor,
    #                         replicate_no=replicate_no,
    #                     )
    #                     filepath = "/home/rohitbokade/repos/pytsc/"
    #                     filepath += "pytsc/scenarios/cityflow/"
    #                     filepath += f"{scenario}/disrupted/"
    #                     filepath += f"r_{disruption_ratio}__"
    #                     filepath += f"p_{speed_reduction_factor}/"
    #                     print(filepath)
    #                     trip_generator.generate_flows(filepath=filepath)
