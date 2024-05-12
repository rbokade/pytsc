from pytsc.backends.cityflow.trip_generator import CityFlowTripGenerator

if __name__ == "__main__":

    for mean_flow in [200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        trip_generator = CityFlowTripGenerator(
            scenario="syn_1x1_gaussian_1h",
            start_time=0,
            end_time=3600,
            inter_mu=3600 / mean_flow,
            inter_sigma=0.80,
            turn_probs=[0.1, 0.3, 0.6],
        )
        trip_generator.generate_flows(
            filepath="/home/rohitbokade/repos/pytsc/pytsc/tests"
        )
