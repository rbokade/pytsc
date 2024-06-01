from pytsc.backends.cityflow.trip_generator import (
    IntervalCityFlowTripGenerator,
)


if __name__ == "__main__":

    segment_duration = 360
    shape = 1.5
    scale = 300
    max_flow_rate = 600

    for size in [1]:
        for replicate_no in range(1, 11, 1):
            trip_generator = IntervalCityFlowTripGenerator(
                scenario=f"syn_{size}x{size}",
                start_time=0,
                end_time=3600,
                inter_mu=3600 / max_flow_rate,
                inter_sigma=0.80,
                turn_probs=[0.1, 0.3, 0.6],
            )
            trip_generator.generate_flows(
                filepath="/home/rohitbokade/repos/pytsc/pytsc/tests",
                replicate_no=replicate_no,
                interval_duration=segment_duration,
                shape=1.5,
                scale=300,
            )
