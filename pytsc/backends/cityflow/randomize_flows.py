import os
import json
import random

scenario_path = "/home/rohitbokade/repos/pytsc/pytsc/scenarios/cityflow/hangzhou_4_4"
file_path = os.path.join(scenario_path, "anon_4_4_hangzhou_real_5816.json")
with open(file_path, "r") as f:
    original_data = json.load(f)

max_time = max(entry["startTime"] for entry in original_data)
randomized_flows_path = os.path.join(scenario_path, "randomized")
os.makedirs(randomized_flows_path, exist_ok=True)
for i in range(100):
    data = [entry.copy() for entry in original_data]
    for entry in data:
        random_offset = random.randint(-5, 5)
        entry["startTime"] = max(0, min(entry["startTime"] + random_offset, max_time))
        entry["endTime"] = max(0, min(entry["endTime"] + random_offset, max_time))
    output_file_path = os.path.join(
        randomized_flows_path, f"anon_4_4_hangzhou_real_5816_{i}.json"
    )
    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Updated JSON saved to {output_file_path}")
