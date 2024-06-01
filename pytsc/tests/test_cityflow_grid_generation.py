from pytsc.backends.cityflow.grid_generator import CityFlowGridNetworkGenerator

if __name__ == "__main__":

    for mean_flow in [600]:
        trip_generator = CityFlowGridNetworkGenerator(
            cityflow_dir="/home/rohitbokade/CityFlow/",
            nrows=1,
            ncols=1,
        )
        trip_generator.generate_roadnet_files()
