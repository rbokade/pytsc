from pytsc.backends.cityflow.trip_generator import CityFlowOneWayTripGenerator

if __name__ == "__main__":

    # Generate hourly flows
    mean_flow_ns = 500
    mean_flow_ew = 500

    trip_generator = CityFlowOneWayTripGenerator(
        scenario="syn_5x5_oneway",
        start_time=0,
        end_time=3600,
        inter_mu_ns=3600 / mean_flow_ns,
        inter_sigma_ns=5.0,
        inter_mu_ew=3600 / mean_flow_ew,
        inter_sigma_ew=5.0,
    )
    trip_generator.generate_flows(
        filepath="/Users/rohitbokade/repos/pytsc/pytsc/tests"
    )
