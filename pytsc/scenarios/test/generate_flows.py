import argparse


def create_vehicles(platoon_arrival_t, platoon_size, ns_veh_per_hr):
    # Calculating the departure interval for NS vehicles in seconds
    ns_departure_interval = 3600 / ns_veh_per_hr

    # NS vehicles for both routes A and B
    vehicles = []
    for i in range(int(120 / ns_departure_interval)):
        depart_time_A = i * ns_departure_interval
        depart_time_B = i * ns_departure_interval
        vehicles.append(
            (
                depart_time_A,
                '<route id="route_ns_'
                + str(i)
                + '_A" edges="top0A0 A0bottom0"/><vehicle id="ns_'
                + str(i)
                + '_A" depart="'
                + str(depart_time_A)
                + '" route="route_ns_'
                + str(i)
                + '_A" departSpeed="max"/>',
            )
        )
        vehicles.append(
            (
                depart_time_B,
                '<route id="route_ns_'
                + str(i)
                + '_B" edges="top1B0 B0bottom1"/><vehicle id="ns_'
                + str(i)
                + '_B" depart="'
                + str(depart_time_B)
                + '" route="route_ns_'
                + str(i)
                + '_B" departSpeed="max"/>',
            )
        )

    # Platoon vehicles - platoon_size vehicles starting at platoon_arrival_t, 1 vehicle per second
    platoon_route_edges = (
        "left0A0 A0B0 B0right0"  # Correct sequence of edges for platoon
    )
    platoon_vehicles = "\n".join(
        [
            '<route id="route_platoon_'
            + str(i)
            + '" edges="'
            + platoon_route_edges
            + '"/><vehicle id="platoon_'
            + str(i)
            + '" depart="'
            + str(platoon_arrival_t + i)
            + '" route="route_platoon_'
            + str(i)
            + '" color="green" departSpeed="max"/>'
            for i in range(platoon_size)
        ]
    )

    # Sorting the vehicles by departure time
    vehicles += [
        (platoon_arrival_t + i, platoon_vehicles.split("\n")[i])
        for i in range(platoon_size)
    ]
    vehicles_sorted = sorted(vehicles, key=lambda x: x[0])

    # Combining the sorted vehicles
    vehicles_combined = (
        """
    <routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
        """
        + "\n".join([v[1] for v in vehicles_sorted])
        + """
    </routes>
    """
    )

    # Writing the modified routes file
    with open("1x2_regular_grid.rou.xml", "w") as file:
        file.write(vehicles_combined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create vehicles for SUMO simulation."
    )
    parser.add_argument(
        "--platoon_arrival_t",
        type=int,
        default=20,
        help="Time step at which the platoon arrives",
    )
    parser.add_argument(
        "--platoon_size",
        type=int,
        default=10,
        help="Number of vehicles in the platoon",
    )
    parser.add_argument(
        "--ns_veh_per_hr",
        type=float,
        default=360,
        help="Flow rate of the NS flow (number of vehicles per hour)",
    )
    args = parser.parse_args()

    create_vehicles(
        args.platoon_arrival_t, args.platoon_size, args.ns_veh_per_hr
    )
