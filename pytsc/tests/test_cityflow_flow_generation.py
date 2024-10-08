from pytsc.backends.cityflow.trip_generator import CityFlowRandomizedTripGenerator

if __name__ == "__main__":

    scenario = "manhattan_28_7"
    trip_generator = CityFlowRandomizedTripGenerator(
        scenario=scenario,
        start_time=0,
        end_time=3600,
    )
    for i in range(100):
        for flow_type in ["low", "medium", "high"]:
            trip_generator.generate_flows(
                filepath=f"/home/rohitbokade/repos/pytsc/pytsc/scenarios/cityflow/{scenario}/randomized",
                flow_type=flow_type,
                replicate_no=i,
            )
