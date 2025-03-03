import os
import subprocess

# Define the network files
files = [
    "cologne_1",
    "cologne_2",
    "cologne_3",
    "cologne_4",
    "cologne_5",
    "cologne_6",
    "cologne_7",
    "cologne_8",
]

# Expand the user path for SUMO's randomTrips.py script
RANDOM_TRIPS_SCRIPT = os.path.expanduser("~/sumo/tools/randomTrips.py")

# List of trip generation command templates for different traffic intensities
COMMANDS = [
    "python {random_trips_script} --begin 0 --end 3600 --period 4 --binomial 4 --net-file {net_file}.net.xml -o {output_file}_light.trips.xml",
    "python {random_trips_script} --begin 0 --end 3600 --period 3 --binomial 4 --net-file {net_file}.net.xml -o {output_file}_medium.trips.xml",
    "python {random_trips_script} --begin 0 --end 3600 --period 2 --binomial 4 --net-file {net_file}.net.xml -o {output_file}_heavy.trips.xml",
]


def run_command(command):
    """Executes a shell command and prints output/errors."""
    try:
        print(f"Executing: {command}")
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")


# Execute trip generation for each network file
for file in files:
    for cmd_template in COMMANDS:
        trip_cmd = cmd_template.format(
            random_trips_script=RANDOM_TRIPS_SCRIPT, net_file=file, output_file=file
        )
        run_command(trip_cmd)

print("Network and trip generation completed!")
