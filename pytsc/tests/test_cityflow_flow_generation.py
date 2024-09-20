from pytsc.backends.cityflow.trip_generator import CityFlowRandomizedTripGenerator

if __name__ == "__main__":

   
    trip_generator = CityFlowRandomizedTripGenerator(
        scenario="jinan_3_4",
        start_time=0,
        end_time=3600,
    )
    for i in range(100):
        trip_generator.generate_flows(
            filepath="/Users/rohitbokade/repos/pytsc/pytsc/scenarios/cityflow/jinan_3_4/randomized",
            replicate_no=i,
        )

