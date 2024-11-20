import os

from pytsc.backends.cityflow.trip_generator import (
    CityFlowTripGenerator,
    LinkDisruptedCityFlowTripGenerator,
)

if __name__ == "__main__":

    scenario = "hangzhou_4_4"
    base_filepath = (
        f"/home/rohitbokade/repos/pytsc/pytsc/scenarios/cityflow/{scenario}/"
    )
    # os.makedirs(os.path.join(filepath, "train/normal/"), exist_ok=True)
    # os.makedirs(os.path.join(filepath, "train/link_disrupted/"), exist_ok=True)
    # os.makedirs(os.path.join(filepath, "test/flow_disrupted/"), exist_ok=True)
    # os.makedirs(os.path.join(filepath, "test/link_disrupted/"), exist_ok=True)

    base_flow_rate = 700

    # for i in range(10):
    #     trip_generator = CityFlowTripGenerator(
    #         scenario=scenario,
    #         inter_mu=3600 / base_flow_rate,
    #         inter_sigma=0.8,
    #         start_time=0,
    #         end_time=3600,
    #     )
    #     trip_generator.generate_flows(
    #         filepath=os.path.join(filepath, "train/normal/"),
    #         replicate_no=i,
    #     )

    for flow_rate in [600, base_flow_rate, 750, 800]:
        filepath = os.path.join(base_filepath, f"train/flow_disrupted/{flow_rate}/")
        os.makedirs(filepath, exist_ok=True)
        for i in range(10):
            trip_generator = CityFlowTripGenerator(
                scenario=scenario,
                inter_mu=3600 / flow_rate,
                inter_sigma=0.8,
                start_time=0,
                end_time=3600,
            )
            trip_generator.generate_flows(filepath=filepath, replicate_no=i)

    for flow_rate in [650, 850, 900]:
        filepath = os.path.join(base_filepath, f"test/flow_disrupted/{flow_rate}/")
        os.makedirs(filepath, exist_ok=True)
        for i in range(10):
            trip_generator = CityFlowTripGenerator(
                scenario=scenario,
                inter_mu=3600 / flow_rate,
                inter_sigma=0.8,
                start_time=0,
                end_time=3600,
            )
            trip_generator.generate_flows(filepath=filepath, replicate_no=i)

    for disruption_ratio in [0.1, 0.3]:
        filepath = os.path.join(
            base_filepath,
            f"train/link_disrupted/{str(disruption_ratio).replace('.', '_')}/",
        )
        os.makedirs(filepath, exist_ok=True)
        for i in range(10):
            link_disrupted_trip_generator = LinkDisruptedCityFlowTripGenerator(
                scenario=scenario,
                inter_mu=3600 / base_flow_rate,
                inter_sigma=0.8,
                start_time=0,
                end_time=3600,
                disruption_ratio=disruption_ratio,
            )
            link_disrupted_trip_generator.generate_flows(
                filepath=filepath, replicate_no=i
            )

    for disruption_ratio in [0.2, 0.4]:
        filepath = os.path.join(
            base_filepath,
            f"test/link_disrupted/{str(disruption_ratio).replace('.', '_')}/",
        )
        os.makedirs(filepath, exist_ok=True)
        for i in range(10):
            link_disrupted_trip_generator = LinkDisruptedCityFlowTripGenerator(
                scenario=scenario,
                inter_mu=3600 / base_flow_rate,
                inter_sigma=0.8,
                start_time=0,
                end_time=3600,
                disruption_ratio=disruption_ratio,
            )
            link_disrupted_trip_generator.generate_flows(
                filepath=filepath, replicate_no=i
            )
