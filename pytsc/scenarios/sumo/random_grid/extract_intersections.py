import xml.etree.ElementTree as ET
import argparse

def extract_subnetwork(root, tl_id, center=False):
    """Extract sub-network for the traffic light `tl_id` (and its incoming lanes) from the parsed XML root."""
    # 1. Identify the traffic light node and any internal junctions within it
    node_ids = {tl_id}
    for j in root.findall('junction'):
        if j.get('type') == 'internal' and j.get('id').startswith(f":{tl_id}_"):
            node_ids.add(j.get('id'))
    # 2. Collect all edges entering or leaving these nodes
    edges = []
    for e in root.findall('edge'):
        if e.get('from') in node_ids or e.get('to') in node_ids:
            edges.append(e)
    # Remove duplicate edges (if any)
    seen = set()
    edges = [e for e in edges if not (e.get('id') in seen or seen.add(e.get('id')))]
    # 3. Include any new nodes from these edges (outer end nodes of incoming/outgoing roads)
    for e in edges:
        node_ids.add(e.get('from')); node_ids.add(e.get('to'))
    # 4. Create a new network XML element
    new_net = ET.Element('net')
    # Copy coordinate system info
    loc = root.find('location')
    new_loc = ET.Element('location')
    if loc is not None:
        for attr, val in loc.attrib.items():
            new_loc.set(attr, val)
    # Prepare coordinate shift if centering is requested
    shift_x = shift_y = 0.0
    if center:
        tl_node = root.find(f"junction[@id='{tl_id}']")
        if tl_node is not None:
            shift_x = float(tl_node.get('x', 0.0))
            shift_y = float(tl_node.get('y', 0.0))
        # Update netOffset in location (add shift to existing offset)
        if new_loc.get('netOffset'):
            off_x, off_y = map(float, new_loc.get('netOffset').split(','))
            new_loc.set('netOffset', f"{off_x + shift_x:.2f},{off_y + shift_y:.2f}")
        else:
            new_loc.set('netOffset', f"{shift_x:.2f},{shift_y:.2f}")
    if not new_loc.get('netOffset'):
        new_loc.set('netOffset', "0.00,0.00")
    # 5. Append edges (incoming, outgoing, internal) to the new network
    edges.sort(key=lambda e: 0 if e.get('function') == 'internal' else 1)
    for e in edges:
        new_net.append(e)  # use deep copy if using the same parsed tree for multiple outputs
    # 6. Append the traffic light logic for this intersection
    tl_logic = root.find(f"tlLogic[@id='{tl_id}']")
    if tl_logic is not None:
        new_net.append(tl_logic)
    # 7. Append relevant junctions (the TL itself, connected nodes, internal nodes)
    for j in root.findall('junction'):
        jid = j.get('id')
        if jid in node_ids:
            new_junc = ET.Element('junction')
            for attr, val in j.attrib.items():
                new_junc.set(attr, val)
            # Downgrade other traffic lights to uncontrolled
            if jid != tl_id and new_junc.get('type') in ('traffic_light', 'traffic_light_right_on_red'):
                new_junc.set('type', 'priority')
            # Apply coordinate shift if centering
            if center and new_junc.get('x') and new_junc.get('y'):
                new_x = float(new_junc.get('x')) - shift_x
                new_y = float(new_junc.get('y')) - shift_y
                new_junc.set('x', f"{new_x:.2f}"); new_junc.set('y', f"{new_y:.2f}")
            new_net.append(new_junc)
    # 8. Append connections for movements within this intersection
    edge_ids = {e.get('id') for e in edges}
    for conn in root.findall('connection'):
        if conn.get('from') in edge_ids and conn.get('to') in edge_ids:
            new_net.append(conn)
    # 9. Update the location’s convBoundary to the sub-network bounds
    xs, ys = [], []
    for j in new_net.findall('junction'):
        if j.get('x') and j.get('y'):
            xs.append(float(j.get('x'))); ys.append(float(j.get('y')))
    if xs and ys:
        new_loc.set('convBoundary', f"{min(xs):.2f},{min(ys):.2f},{max(xs):.2f},{max(ys):.2f}")
    # Insert the location element at the top
    new_net.insert(0, new_loc)
    return new_net

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract single-intersection sub-networks from a SUMO .net.xml")
    parser.add_argument("-i", "--input", required=True, help="Path to the input .net.xml file")
    parser.add_argument("-s", "--scenario", required=True, help="Scenario name prefix for output files")
    parser.add_argument("--center", action="store_true", help="Recenter coordinates (place traffic light at (0,0) in output)")
    args = parser.parse_args()
    # Parse the input network
    tree = ET.parse(args.input)
    root = tree.getroot()
    # Find all traffic light junction IDs
    tl_ids = [j.get('id') for j in root.findall('junction')
              if j.get('type') in ('traffic_light', 'traffic_light_right_on_red')]
    # Generate and save a sub-network for each traffic light
    for tl_id in tl_ids:
        sub_net = extract_subnetwork(root, tl_id, center=args.center)
        output_path = f"{args.scenario}_{tl_id}.net.xml"
        ET.ElementTree(sub_net).write(output_path, encoding="UTF-8", xml_declaration=True)
        print(f"Saved sub-network for traffic light '{tl_id}' as {output_path}")

