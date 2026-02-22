"""AI state representation utilities for the simulator.

Provides a compact state vector S describing current traffic per direction
and per-lane densities so higher-level components (reasoner, predictor)
can use it.
"""
from typing import Dict, Any

def compute_state(vehicles: Dict, direction_numbers: dict, no_of_lanes: int = 3) -> Dict[str, Any]:
    """Compute a simple state representation from the simulator's `vehicles` data.

    Returns a dict with:
      - queue_lengths: list of queued vehicles (crossed==0) per direction
      - lane_counts: nested list [[lane0_count, lane1_count, ...], ...] per direction
      - total_vehicles: total queued vehicles
    """
    queue_lengths = []
    lane_counts = []
    total = 0
    for i in range(len(direction_numbers)):
        dname = direction_numbers[i]
        lanes = []
        qlen = 0
        for lane in range(no_of_lanes):
            lane_list = vehicles[dname].get(lane, [])
            # count vehicles that haven't crossed
            c = sum(1 for v in lane_list if getattr(v, 'crossed', 0) == 0)
            lanes.append(c)
            qlen += c
        lane_counts.append(lanes)
        queue_lengths.append(qlen)
        total += qlen

    state = {
        'queue_lengths': queue_lengths,
        'lane_counts': lane_counts,
        'total_vehicles': total,
    }
    return state
