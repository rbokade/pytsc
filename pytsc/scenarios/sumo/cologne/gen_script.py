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

# Path to SUMO's randomTrips.py script
RANDOM_TRIPS_SCRIPT = "~/sumo/tools/randomTrips.py"

# Trip generation parameters
TRIPGEN_CMD_TEMPLATE = (
    "python {random_trips_script} --begin 0 --end 3600 --period 2.0 --net-file {net_file}.net.xml -o {output_file}.trips.xml"
)

def run_command(command):
    """Executes a shell command and prints output/errors."""
    try:
        print(f"Executing: {command}")
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

# Execute trip generation
for file in files:
    trip_cmd = TRIPGEN_CMD_TEMPLATE.format(
        random_trips_script=RANDOM_TRIPS_SCRIPT,
        net_file=file,
        output_file=file,
    )
    run_command(trip_cmd)

print("Network and trip generation completed!")
