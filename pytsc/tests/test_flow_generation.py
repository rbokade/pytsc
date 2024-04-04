from pytsc.backends.cityflow.trip_generator import CityFlowTripGenerator

if __name__ == "__main__":

    trip_generator = CityFlowTripGenerator(
        scenario="syn_2x2_gaussian_500_1h",
        start_time=0,
        end_time=3600,
        inter_mu=5.0,
        inter_sigma=0.75,
        turn_probs=[0.1, 0.3, 0.6],
    )
    trip_generator.generate_flows(
        filepath="/home/rohitbokade/repos/pytsc/pytsc/tests"
    )
