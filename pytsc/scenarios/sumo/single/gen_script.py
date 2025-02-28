import subprocess

# Define the network files
files = [
    "single",
    "single_top",
    "single_right",
    "single_left",
    "single_bottom",
]

# Path to SUMO's randomTrips.py script
RANDOM_TRIPS_SCRIPT = "~/sumo/tools/randomTrips.py"

# Network generation parameters
NETGENERATE_CMD_TEMPLATE = (
    "netgenerate --grid --grid.x-number 1 --grid.y-number 1 "
    "--grid.x-length 150 --grid.y-length 150 --grid.attach-length 150 "
    "--default.lanenumber 2 --tls.guess --tls.layout 'incoming' "
    "--output-file {output_file}.net.xml"
)

# Trip generation parameters
TRIPGEN_CMD_TEMPLATE = (
    "python {random_trips_script} --begin 0 --end 360 --random-depart "
    "--fringe-factor 100 --period 1 --net-file {net_file}.net.xml -o {output_file}.trips.xml"
)

def run_command(command):
    """Executes a shell command and prints output/errors."""
    try:
        print(f"Executing: {command}")
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

# Execute network generation
# for file in files:
#     net_cmd = NETGENERATE_CMD_TEMPLATE.format(output_file=file)
#     run_command(net_cmd)

# Execute trip generation
for file in files:
    trip_cmd = TRIPGEN_CMD_TEMPLATE.format(
        random_trips_script=RANDOM_TRIPS_SCRIPT,
        net_file=file,
        output_file=file,
    )
    run_command(trip_cmd)

print("Network and trip generation completed!")
