import os
import subprocess

# Define the network files
files = [f"random_grid_{i}" for i in range(1, 11)]

# Expand the user path for SUMO's randomTrips.py script
RANDOM_TRIPS_SCRIPT = os.path.expanduser("~/sumo/tools/randomTrips.py")

# List of trip generation command templates for different traffic intensities,
# along with their respective flow type suffix.
COMMANDS = [
    ("light", "python {random_trips_script} --begin 0 --end 3600 --period 4 --net-file {net_file}.net.xml -o {output_file}_light.trips.xml"),
    ("medium", "python {random_trips_script} --begin 0 --end 3600 --period 3 --net-file {net_file}.net.xml -o {output_file}_medium.trips.xml"),
    ("heavy", "python {random_trips_script} --begin 0 --end 3600 --period 2 --net-file {net_file}.net.xml -o {output_file}_heavy.trips.xml"),
]

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

# Execute trip generation and create .sumocfg files for each network file and traffic intensity
for file in files:
    for flow_type, cmd_template in COMMANDS:
        trip_cmd = cmd_template.format(
            random_trips_script=RANDOM_TRIPS_SCRIPT,
            net_file=file,
            output_file=file
        )
        run_command(trip_cmd)
        create_sumocfg(file, flow_type)

print("Network, trip generation, and .sumocfg file creation completed!")
