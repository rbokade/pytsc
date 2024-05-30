from pytsc.backends.cityflow.trip_generator import CityFlowTripGenerator

if __name__ == "__main__":

    for mean_flow in [500, 525, 550, 575, 600, 625, 650, 675, 700]:
        trip_generator = CityFlowTripGenerator(
            scenario="syn_5x5",
            start_time=0,
            end_time=3600,
            inter_mu=3600 / mean_flow,
            inter_sigma=0.80,
            turn_probs=[0.1, 0.3, 0.6],
        )
        trip_generator.generate_flows(
            filepath="/home/rohitbokade/repos/pytsc/pytsc/tests"
        )
