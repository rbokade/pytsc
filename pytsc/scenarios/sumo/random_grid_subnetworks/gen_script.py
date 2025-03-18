import os
import subprocess

# Define the SUMO randomTrips.py script path
RANDOM_TRIPS_SCRIPT = os.path.expanduser("~/sumo/tools/randomTrips.py")

# Define subnetwork details with computed period values
subnetwork_periods = {
    "random_subnetwork_1": {
        "light": 2.582792860012627,
        "medium": 2.0662342880101017,
        "heavy": 1.5496757160075763,
    },
    "random_subnetwork_2": {
        "light": 2.364418485042247,
        "medium": 1.8915347880337976,
        "heavy": 1.4186510910253483,
    },
    "random_subnetwork_3": {
        "light": 2.582792860012627,
        "medium": 2.0662342880101017,
        "heavy": 1.5496757160075763,
    },
    "random_subnetwork_4": {
        "light": 2.582792860012627,
        "medium": 2.0662342880101017,
        "heavy": 1.5496757160075763,
    },
    "random_subnetwork_5": {
        "light": 2.567739160596715,
        "medium": 2.054191328477372,
        "heavy": 1.5406434963580289,
    },
}

# Define the trip generation command template
TRIP_CMD_TEMPLATE = (
    "python {random_trips_script} --begin 0 --end 3600 --period {period} "
    "--binomial 10 --random-depart --fringe-factor 100 --validate --remove-loops "
    "--min-distance 600 --net-file {net_file}.net.xml -o {output_file}.trips.xml"
)


def run_command(command):
    """Executes a shell command and prints output/errors."""
    try:
        print(f"Executing: {command}")
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")


def create_sumocfg(net_file, flow_type):
    """Creates a .sumocfg file for the given network file and flow type."""
    sumocfg_content = f"""<?xml version='1.0' encoding='UTF-8'?>
<configuration>
  <input>
    <net-file value="{net_file}.net.xml"/>
    <route-files value="{net_file}_{flow_type}.trips.xml"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="3600"/>
  </time>
  <report>
    <verbose value="true"/>
    <no-step-log value="true"/>
  </report>
</configuration>
"""
    sumocfg_filename = f"{net_file}_{flow_type}.sumocfg"
    with open(sumocfg_filename, "w") as f:
        f.write(sumocfg_content)
    print(f"Created {sumocfg_filename}")


# Execute trip generation and create .sumocfg files for each subnetwork
for net_file, periods in subnetwork_periods.items():
    for flow_type, period in periods.items():
        # Format the trip generation command with the correct period
        trip_cmd = TRIP_CMD_TEMPLATE.format(
            random_trips_script=RANDOM_TRIPS_SCRIPT,
            net_file=net_file,
            output_file=f"{net_file}_{flow_type}",
            period=period,
        )
        run_command(trip_cmd)

        # Create corresponding .sumocfg file
        create_sumocfg(net_file, flow_type)

print("Network trip generation and .sumocfg file creation completed!")
