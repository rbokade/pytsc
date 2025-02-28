import subprocess

# Define output filenames
NET_FILE = "random.net.xml"
TRIP_FILE = "random.trips.xml"

# Path to SUMO's randomTrips.py script
RANDOM_TRIPS_SCRIPT = "~/sumo/tools/randomTrips.py"


def run_command(command):
    """Executes a shell command and prints output/errors."""
    try:
        print(f"Executing: {command}")
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")


# 1. Generate a Random Network
netgenerate_cmd = (
    f"netgenerate --rand --rand.iterations 200 --rand.connectivity 0.75 "
    f"--rand.grid --rand.min-angle 90 --rand.min-distance 200 --rand.max-distance 500 "
    f"--tls.guess True --default.lanenumber 2 --fringe.guess --output-file {NET_FILE}"
)
run_command(netgenerate_cmd)

# 2. Generate Random Trips with Insertion Rate
tripgen_cmd = (
    f"python {RANDOM_TRIPS_SCRIPT} --begin 0 --end 3600 --insertion-rate 8200 "
    f"--random-depart --min-distance.fringe 1200 --validate --remove-loops "
    f"--net-file {NET_FILE} -o {TRIP_FILE}"
)
run_command(tripgen_cmd)

print("SUMO network and trip generation completed successfully!")
