from pytsc.backends.cityflow.trip_generator import (
    IntervalCityFlowTripGenerator,
)


if __name__ == "__main__":

    segment_duration = 360
    scale = 100
    max_flow_rate = 700
    shapes = [1.5, 2.5]

    for size in [(2, 2)]:
        for replicate_no in range(1, 6, 1):
            for shape in shapes:
                trip_generator = IntervalCityFlowTripGenerator(
                    scenario=f"syn_{size[0]}x{size[1]}",
                    start_time=0,
                    end_time=3600,
                    inter_mu=3600 / max_flow_rate,
                    inter_sigma=0.80,
                    turn_probs=[0.1, 0.3, 0.6],
                )
                trip_generator.generate_flows(
                    filepath="/Users/rohitbokade/repos/pytsc/pytsc/tests",
                    replicate_no=replicate_no,
                    interval_duration=segment_duration,
                    shape=shape,
                    scale=scale,
                )
