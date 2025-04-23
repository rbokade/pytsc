import argparse
import json
import xml.etree.ElementTree as ET

DEFAULT_VEHICLE_ATTRIBUTES = {
    "length": 5.0,
    "width": 2.0,
    "maxPosAcc": 2.0,
    "maxNegAcc": 4.5,
    "usualPosAcc": 2.0,
    "usualNegAcc": 4.5,
    "minGap": 2.5,
    "maxSpeed": 13.89,
    "headwayTime": 2,
}


def convert_sumo_to_cityflow(sumo_trip_file_path, output_file_path):
    sumo_tree = ET.parse(sumo_trip_file_path)
    sumo_root = sumo_tree.getroot()
    sumo_trips = []
    for vehicle in sumo_root.findall("vehicle"):
        trip_info = {
            "id": vehicle.get("id"),
            "depart": vehicle.get("depart"),
            "route": vehicle.find("route").get("edges").split(),
        }
        sumo_trips.append(trip_info)
    cityflow_trips = []
    for trip in sumo_trips:
        cityflow_trip = {
            "vehicle": DEFAULT_VEHICLE_ATTRIBUTES,
            "route": trip["route"],
            "interval": 1.0,
            "startTime": round(float(trip["depart"])),
            "endTime": round(float(trip["depart"])),
        }
        cityflow_trips.append(cityflow_trip)
    with open(output_file_path, "w") as file:
        json.dump(cityflow_trips, file, indent=2)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sumo_routes_file",
        help="SUMO routes file",
        type=str,
    )
    parser.add_argument(
        "--cityflow_routes_file",
        help="CityFlow routes file",
        type=str,
    )
    args = parser.parse_args()

    convert_sumo_to_cityflow(args.sumo_routes_file, args.cityflow_routes_file)
