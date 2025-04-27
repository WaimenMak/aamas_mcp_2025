# -*- coding: utf-8 -*-
# @Time    : 27/04/2025 10:16
# @Author  : mmai
# @FileName: utils
# @Software: PyCharm
from collections import defaultdict


def simulate_schedule_cost_allocated_shared_arrival(vessel, vessel_schedule_copy, start_time, headquarters=None, payments=None):
    """
    Simulates a vessel's schedule, allocating travel costs to a port among
    trades on board AND trades involved in immediate events at that port.

    Output: (Same as previous version, but with refined cost allocation)
    total_cost: float - The overall schedule cost (travel + operation + idle)
    trade_specific_costs: dict {trade_object: cost}
    total_idle_cost: float
    total_idle_time: float
    is_feasible: bool
    pick_up_times: dict {trade_object: time}
    drop_off_times: dict {trade_object: time}
    """
    trade_specific_costs = defaultdict(float)
    total_idle_time = 0
    total_travel_cost = 0
    total_operation_cost = 0
    pick_up_times = {}
    drop_off_times = {}
    is_feasible = True

    if not vessel_schedule_copy:
        horizon_duration = 720
        total_idle_time = horizon_duration
        total_idle_cost = vessel.get_idle_consumption(total_idle_time)
        total_cost = total_idle_cost # Only idle cost if schedule is empty
        return total_cost, trade_specific_costs, total_idle_time, pick_up_times, drop_off_times

    vessel_schedule = vessel_schedule_copy.get_simple_schedule()
    vessel_trades = vessel_schedule_copy.get_scheduled_trades()

    current_time = float(start_time)
    current_port = vessel.location
    trades_on_board = set()

    # Optional sorting (same as before) - important for grouping logic
    # try:
    #     # Add handling for None time_window during sorting
    #     def sort_key(item):
    #         event_type, trade = item
    #         if trade.time_window is None:
    #             # If no time window, treat as unbounded for sorting
    #             return (float('-inf'), float('inf'))
    #         if event_type == 'PICK_UP':
    #             return (trade.time_window[0], trade.time_window[1])
    #         else: # DROP_OFF
    #             return (trade.time_window[2], trade.time_window[3])
    #
    #     vessel_schedule.sort(key=sort_key)
    # except AttributeError:
    #     print("Warning: Could not sort vessel schedule based on time windows (AttributeError).")
    #     pass
    # except IndexError:
    #     print("Warning: Could not sort vessel schedule based on time windows (IndexError - possibly incomplete time_window).")
    #     pass

    processed_indices = set()
    current_schedule_index = 0

    while current_schedule_index < len(vessel_schedule):
        if current_schedule_index in processed_indices:
            current_schedule_index += 1
            continue

        # --- Identify the next block of events at the same target port ---
        first_event_index = current_schedule_index
        # Check if index is valid before accessing
        if first_event_index >= len(vessel_schedule):
             break # Should not happen if loop condition is correct, but safety check
        first_event_type = vessel_schedule[first_event_index][0]
        first_trade = vessel_schedule[first_event_index][1]

        if first_event_type == 'PICK_UP':
            target_port = first_trade.origin_port
        elif first_event_type == 'DROP_OFF':
            target_port = first_trade.destination_port
        else:
            current_schedule_index += 1
            continue

        # Find all consecutive events at this target_port
        events_at_target_port_indices = []
        trades_involved_at_target = set()
        temp_idx = first_event_index
        while temp_idx < len(vessel_schedule):
            # Check if index is valid before accessing
            if temp_idx >= len(vessel_schedule):
                break
            evt_type, trd = vessel_schedule[temp_idx]
            port_for_this_event = trd.origin_port if evt_type == 'PICK_UP' else trd.destination_port
            if port_for_this_event == target_port:
                events_at_target_port_indices.append(temp_idx)
                trades_involved_at_target.add(trd)
                temp_idx += 1
            else:
                break # Stop when the port changes

        # --- 1. Travel to the target port ---
        segment_travel_cost = 0
        travel_time = 0 # Initialize travel_time
        responsible_trades = set() # Initialize responsible_trades here as an empty set

        if current_port != target_port:
            travel_distance = headquarters.get_network_distance(current_port, target_port)
            if travel_distance is None or travel_distance == float('inf'):
                 print(f"Error: Unreachable route from {current_port} to {target_port}")
                 is_feasible = False
                 break

            travel_time = vessel.get_travel_time(travel_distance) # Use ceil for safety

            # Determine if travel is ballast or laden based on state *before* travel
            is_ballast = len(trades_on_board) == 0
            if is_ballast:
                segment_travel_cost = vessel.get_ballast_consumption(travel_time, vessel.speed)
            else:
                segment_travel_cost = vessel.get_laden_consumption(travel_time, vessel.speed)

            total_travel_cost += segment_travel_cost # Add to total travel cost

            # --- Allocate travel cost ---
            # Responsible trades = trades on board + trades involved in events at destination
            responsible_trades = trades_on_board.copy()
            responsible_trades.update(trades_involved_at_target) # Add trades involved at the port

            if responsible_trades: # Avoid division by zero if set is empty
                cost_share = segment_travel_cost / len(responsible_trades)
                for t_resp in responsible_trades:
                    trade_specific_costs[t_resp] += cost_share
            elif segment_travel_cost > 0:
                 # Travel cost occurred but no trades identified as responsible? Log warning.
                 print(f"Warning: Travel cost {segment_travel_cost} to {target_port} not allocated to any trade.")

            # Update current time after travel
            current_time += travel_time

        # --- 2. Process Events at the Target Port ---
        for event_idx in events_at_target_port_indices:
            if event_idx in processed_indices: # Should not happen with current logic, but safe check
                 continue

            event_type = vessel_schedule[event_idx][0]
            trade = vessel_schedule[event_idx][1]

            # --- Determine event time window, handling None and potential errors ---
            earliest_event_time = float('-inf') # Default: No earliest restriction
            latest_event_time = float('inf')   # Default: No latest restriction

            if trade.time_window is not None:
                try:
                    if event_type == 'PICK_UP':
                        # Get potential earliest and latest times
                        e_time = trade.time_window[0]
                        l_time = trade.time_window[1]
                        # Assign actual values, defaulting to infinity if None
                        earliest_event_time = float('-inf') if e_time is None else e_time
                        latest_event_time = float('inf') if l_time is None else l_time
                    elif event_type == 'DROP_OFF':
                        # Get potential earliest and latest times
                        e_time = trade.time_window[2]
                        l_time = trade.time_window[3]
                        # Assign actual values, defaulting to infinity if None
                        earliest_event_time = float('-inf') if e_time is None else e_time
                        latest_event_time = float('inf') if l_time is None else l_time
                    # else: # Optional: Handle unexpected event types if necessary
                    #    print(f"Warning: Unknown event type '{event_type}' encountered.")

                except IndexError:
                    # Handles cases where time_window is too short (e.g., only has 2 elements)
                    print(f"Warning: Incomplete time_window for trade {trade.origin_port} (Event: {event_type}). Treating as unbounded.")
                    # Keep the default infinite bounds assigned earlier
            # else: # trade.time_window is None
                 # Keep the default infinite bounds assigned earlier

            # Check Time Window and Calculate Idle Time for this specific event
            # Note: If latest_event_time is inf, this check will always pass.
            if current_time > latest_event_time:
                print(f"Infeasible: Arrived/Ready at {target_port} for {event_type} of trade at {current_time}, latest allowed is {latest_event_time}")
                is_feasible = False
                break # Break inner loop
                # current_time = latest_event_time # Set to latest event time to avoid infeasibility

            idle_this_segment = 0
            # Note: If earliest_event_time is -inf, this check will likely be false.
            if current_time < earliest_event_time:
                idle_this_segment = earliest_event_time - current_time
                current_time = earliest_event_time

            total_idle_time += idle_this_segment


            # Record the actual event time
            if event_type == 'PICK_UP':
                pick_up_times[trade] = current_time
            else:
                drop_off_times[trade] = current_time

            # Process the specific Event (Loading/Unloading)
            operation_time = 0
            operation_cost = 0
            if event_type == 'PICK_UP':
                if trade in trades_on_board:
                     print(f"Warning: Attempting to pick up trade {trade.id} which is already on board at {target_port}.")
                else:
                     operation_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
                     operation_cost = vessel.get_loading_consumption(operation_time)
                     trade_specific_costs[trade] += operation_cost
                     current_time += operation_time
                     trades_on_board.add(trade)

            elif event_type == 'DROP_OFF':
                 if trade not in trades_on_board:
                      print(f"Warning: Attempting to drop off trade {trade.origin_port} which is not on board at {target_port}.")
                 else:
                      operation_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
                      operation_cost = vessel.get_unloading_consumption(operation_time)
                      trade_specific_costs[trade] += operation_cost
                      current_time += operation_time
                      trades_on_board.remove(trade)

            total_operation_cost += operation_cost # Add to total operation cost
            processed_indices.add(event_idx) # Mark this specific event as done

        if not is_feasible: # If infeasibility occurred while processing events at port
             break # Break outer loop

        # Split the idle cost among the responsible trades
        if responsible_trades:
            idle_cost_share = vessel.get_idle_consumption(total_idle_time) / len(responsible_trades)
            for t_resp in responsible_trades:
                trade_specific_costs[t_resp] += idle_cost_share

        # Update the current port for the next iteration
        current_port = target_port
        # Move the main index past the block we just processed
        current_schedule_index = temp_idx # temp_idx is the index of the first event at the *next* port

    # --- 3. Calculate Final Idle Time & Total Cost ---
    horizon_duration = 720
    end_time = float(start_time) + horizon_duration
    if is_feasible and current_time < end_time: # Only add final idle if feasible
        total_idle_time += end_time - current_time

    total_idle_cost = vessel.get_idle_consumption(total_idle_time)

    # Calculate total cost = travel + operation + idle
    total_cost = total_travel_cost + total_operation_cost + total_idle_cost
    if payments is not None:
        for trade in vessel_trades:
            total_cost -= payments[trade]

    # Return results
    return total_cost, trade_specific_costs, total_idle_time, pick_up_times, drop_off_times
