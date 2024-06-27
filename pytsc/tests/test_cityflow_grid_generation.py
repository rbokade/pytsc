from pytsc.backends.cityflow.grid_generator import (
    CityFlowGridNetworkGenerator,
    CityFlowOneWayGridNetworkGenerator,
)

if __name__ == "__main__":

    grid_generator = CityFlowGridNetworkGenerator(
        cityflow_dir="/Users/rohitbokade/CityFlow/",
        nrows=1,
        ncols=7,
    )
    grid_generator.generate_roadnet_files()

    # grid_generator = CityFlowOneWayGridNetworkGenerator(
    #     cityflow_dir="/Users/rohitbokade/CityFlow/",
    #     nrows=1,
    #     ncols=7,
    # )
    # grid_generator.generate_roadnet_files()
