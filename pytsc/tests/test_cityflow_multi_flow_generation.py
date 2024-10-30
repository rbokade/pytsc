import os

from pytsc.backends.cityflow.trip_generator import (
    CityFlowTripGenerator,
    LinkDisruptedCityFlowTripGenerator,
)

if __name__ == "__main__":

    scenario = "hangzhou_4_4"
    filepath = f"/home/rohitbokade/repos/pytsc/pytsc/scenarios/cityflow/{scenario}/"
    os.makedirs(os.path.join(filepath, "train/normal/"), exist_ok=True)
    os.makedirs(os.path.join(filepath, "train/flow_disrupted/"), exist_ok=True)
    os.makedirs(os.path.join(filepath, "train/link_disrupted/"), exist_ok=True)
    os.makedirs(os.path.join(filepath, "test/flow_disrupted/"), exist_ok=True)
    os.makedirs(os.path.join(filepath, "test/link_disrupted/"), exist_ok=True)

    base_flow_rate = 700

    for i in range(10):
        trip_generator = CityFlowTripGenerator(
            scenario=scenario,
            inter_mu=3600 / base_flow_rate,
            inter_sigma=0.8,
            start_time=0,
            end_time=3600,
        )
        trip_generator.generate_flows(
            filepath=os.path.join(filepath, "train/normal/"),
            replicate_no=i,
        )

    for flow_rate in [600, 750, 800]:
        for i in range(10):
            trip_generator = CityFlowTripGenerator(
                scenario=scenario,
                inter_mu=3600 / flow_rate,
                inter_sigma=0.8,
                start_time=0,
                end_time=3600,
            )
            trip_generator.generate_flows(
                filepath=os.path.join(filepath, "train/flow_disrupted/"),
                replicate_no=i,
            )

    for flow_rate in [650, 850, 900]:
        for i in range(10):
            trip_generator = CityFlowTripGenerator(
                scenario=scenario,
                inter_mu=3600 / flow_rate,
                inter_sigma=0.8,
                start_time=0,
                end_time=3600,
            )
            trip_generator.generate_flows(
                filepath=os.path.join(filepath, "test/flow_disrupted/"),
                replicate_no=i,
            )

    for disruption_ratio in [0.1, 0.3]:
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
                filepath=os.path.join(filepath, "train/link_disrupted/"),
                replicate_no=i,
            )

    for disruption_ratio in [0.2, 0.4]:
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
                filepath=os.path.join(filepath, "test/link_disrupted/"),
                replicate_no=i,
            )
