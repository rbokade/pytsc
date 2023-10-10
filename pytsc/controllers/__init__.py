class BasePhaseSelector:
    def __init__(self, traffic_signal, **kwargs):
        self.traffic_signal = traffic_signal
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.id})"


class FixedTimePhaseSelector(BasePhaseSelector):
    def __init__(self, traffic_signal, green_time=25):
        super(FixedTimePhaseSelector, self).__init__(
            traffic_signal, green_time=green_time
        )
        controller = traffic_signal.controller
        for green_phase_index in traffic_signal.green_phase_indices:
            phase = traffic_signal.phases[green_phase_index]
            controller.phases_min_max_times[phase]["min_time"] = green_time
            controller.phases_min_max_times[phase]["max_time"] = green_time

    def get_action(self, inp, action_mask):
        """
        inp (dict): {"current_phase_index": int, "time_on_phase": int}
        """
        current_phase_index = inp["current_phase_index"]
        time_on_phase = inp["time_on_phase"]
        green_phase_indices = self.traffic_signal.green_phase_indices
        yellow_phase_indices = self.traffic_signal.yellow_phase_indices
        if current_phase_index in green_phase_indices:
            if time_on_phase < self.green_time:
                action = current_phase_index
            else:
                green_index = green_phase_indices.index(current_phase_index)
                action = yellow_phase_indices[green_index]
        else:
            action = action_mask.index(1)
        return action


class GreedyPhaseSelector(BasePhaseSelector):
    def __init__(self, traffic_signal):
        super(GreedyPhaseSelector, self).__init__(traffic_signal)

    def get_action(self, inp, action_mask):
        queues = []
        for action, is_available in enumerate(action_mask):
            if is_available:
                queue = self._compute_queue_for_phase(inp, action)
            else:
                queue = float("-inf")
            queues.append((queue, action))
        _, best_action = max(queues)
        return best_action

    def _compute_queue_for_phase(self, inp, phase_index):
        inc_vehicles = 0
        phase = self.traffic_signal.phases[phase_index]
        phase_inc_out_lanes = self.traffic_signal.phase_to_inc_out_lanes[phase]
        for inc_lane, _ in phase_inc_out_lanes:
            inc_vehicles += inp["lane"][inc_lane]["n_queued"]
        return inc_vehicles


class MaxPressurePhaseSelector(BasePhaseSelector):
    def __init__(self, traffic_signal):
        super(MaxPressurePhaseSelector, self).__init__(traffic_signal)

    def get_action(self, inp, action_mask):
        pressures = []
        for action, is_available in enumerate(action_mask):
            if is_available:
                pressure = self._compute_pressure_for_phase(inp, action)
            else:
                pressure = float("-inf")
            pressures.append((pressure, action))
        _, best_action = max(pressures)
        return best_action

    def _compute_pressure_for_phase(self, inp, phase_index):
        """
        inp (dict): network.simulator.step_measurements["lane"]
        """
        pressure = 0
        phase = self.traffic_signal.phases[phase_index]
        phase_inc_out_lanes = self.traffic_signal.phase_to_inc_out_lanes[phase]
        for inc_lane, out_lane in phase_inc_out_lanes:
            inc_lane_vehicles = inp["lane"][inc_lane]["n_vehicles"]
            out_lane_vehicles = inp["lane"][out_lane]["n_vehicles"]
            pressure += inc_lane_vehicles - out_lane_vehicles
        return pressure


class SOTLPhaseSelector(BasePhaseSelector):
    """
    phi: minimum flow threshold
    mu: maximum gap threshold
    omega: maximum time on phase
    """

    def __init__(self, traffic_signal, phi=3, mu=4, omega=30):
        super(SOTLPhaseSelector, self).__init__(
            traffic_signal, phi=phi, mu=mu, omega=omega
        )
        self.last_vehicle_time = {}

    def get_action(self, inp, action_mask):
        if (
            inp["current_phase_index"]
            in self.traffic_signal.green_phase_indices
        ):
            if inp["time_on_phase"] > self.omega:
                return self._get_next_phase(
                    action_mask, inp["current_phase_index"]
                )
            max_gap = self._compute_max_gap_for_phase(
                inp, inp["current_phase_index"]
            )
            if max_gap > self.mu:
                return self._get_next_phase(
                    action_mask, inp["current_phase_index"]
                )
            if (
                self._compute_flow_for_phase(inp, inp["current_phase_index"])
                < self.phi
            ):
                return self._get_next_phase(
                    action_mask, inp["current_phase_index"]
                )
            return inp["current_phase_index"]
        else:
            return action_mask.index(1)

    def _get_next_phase(self, action_mask, current_phase_index):
        available_phases = [
            i for i, available in enumerate(action_mask) if available
        ]
        if current_phase_index not in available_phases:
            return action_mask.index(1)
        current_index_position = available_phases.index(current_phase_index)
        next_phase_index = (current_index_position + 1) % len(available_phases)
        return available_phases[next_phase_index]

    def _compute_flow_for_phase(self, inp, phase_index):
        total_vehicles = 0
        phase = self.traffic_signal.phases[phase_index]
        phase_inc_out_lanes = self.traffic_signal.phase_to_inc_out_lanes[phase]
        for inc_lane, _ in phase_inc_out_lanes:
            total_vehicles += inp["lane"][inc_lane]["n_vehicles"]
        return total_vehicles

    def _compute_max_gap_for_phase(self, inp, phase_index):
        max_gap = 0
        phase = self.traffic_signal.phases[phase_index]
        phase_inc_out_lanes = self.traffic_signal.phase_to_inc_out_lanes[phase]
        current_time = inp["time"]
        for inc_lane, _ in phase_inc_out_lanes:
            last_vehicle_time = self.last_vehicle_time.get(
                inc_lane, current_time
            )
            gap = current_time - last_vehicle_time
            max_gap = max(max_gap, gap)
            if inp["lane"][inc_lane]["n_vehicles"] > 0:
                self.last_vehicle_time[inc_lane] = current_time
        return max_gap
