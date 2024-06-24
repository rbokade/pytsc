from pytsc.backends.cityflow.grid_generator import (
    CityFlowGridNetworkGenerator,
    CityFlowOneWayGridNetworkGenerator,
)

if __name__ == "__main__":

    # for mean_flow in [600]:
    #     grid_generator = CityFlowGridNetworkGenerator(
    #         cityflow_dir="/home/rohitbokade/CityFlow/",
    #         nrows=1,
    #         ncols=1,
    #     )
    #     grid_generator.generate_roadnet_files()

    grid_generator = CityFlowOneWayGridNetworkGenerator(
        cityflow_dir="/Users/rohitbokade/CityFlow/",
        nrows=5,
        ncols=5,
    )
    grid_generator.generate_roadnet_files()
