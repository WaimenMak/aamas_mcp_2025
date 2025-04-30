# -*- coding: utf-8 -*-
# @Time    : 28/04/2025 16:40
# @Author  : mmai
# @FileName: submmision
# @Software: PyCharm


from mable.cargo_bidding import TradingCompany
from mable.examples.companies import ScheduleProposal
import attrs
from marshmallow import fields
import time
import random
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


def simulate_schedule_cost(vessel, vessel_schedule_copy, start_time, headquarters=None, payments=None):
    """
    Input:
    vessel: vessel object
    vesseel_schedule a list of tuples (type, timewindowtrade)
    pick_up_time: a dictionary of the pick up time of the trades
    drop_off_time: a dictionary of the drop off time of the trades
    headquarters: the headquarters object
    payments: a dictionary of the payments of the trades

    Output:
    cost: the cost of the schedule
    idle_time: the idle time of the vessel
    pick_up_time: the pick up time of the trades
    drop_off_time: the drop off time of the trades
    """
    cost = 0
    current_hold_cargo = 0 # check if the vessel is holding cargo
    # start_time = vessel_schedule[0][1].time
    current_time = start_time
    idle_time = 0
    pick_up_time  = {}
    drop_off_time = {}

    if len(vessel_schedule_copy) == 0:
        idle_time += 720

    vessel_schedule = vessel_schedule_copy.get_simple_schedule()
    vessel_trades = vessel_schedule_copy.get_scheduled_trades()
    for i in range(len(vessel_schedule)):
        # handle None time_window
        if vessel_schedule[i][1].time_window[0] is None:
            earliest_pick_up_time = float('-inf')
        if vessel_schedule[i][1].time_window[1] is None:
            latest_pick_up_time = float('inf')
        if vessel_schedule[i][1].time_window[2] is None:
            earliest_drop_off_time = float('-inf')
        if vessel_schedule[i][1].time_window[3] is None:
            latest_drop_off_time = float('inf')
        else:
            earliest_pick_up_time = vessel_schedule[i][1].time_window[0]
            latest_pick_up_time = vessel_schedule[i][1].time_window[1]
            earliest_drop_off_time = vessel_schedule[i][1].time_window[2]
            latest_drop_off_time = vessel_schedule[i][1].time_window[3]
        if i == 0:
            first_travel_distance = headquarters.get_network_distance(vessel.location, vessel_schedule[i][1].origin_port)
            travel_time = vessel.get_travel_time(first_travel_distance)
            current_time += travel_time
            # check whether the vessel can reach on time
            if current_time > latest_pick_up_time:  # the latest pick up time
                return float('inf'), idle_time, pick_up_time, drop_off_time
            if current_time < earliest_pick_up_time:  # earlier than the earliest pick up time
                idle_time += earliest_pick_up_time - current_time
                current_time = earliest_pick_up_time # update the current time


            ballast_cost = vessel.get_ballast_consumption(travel_time, vessel.speed)
            cost += ballast_cost
            # record the pick up time
            # pick_up_time = pick_up_time.get(vessel_schedule[i][1], current_time)
            pick_up_time[vessel_schedule[i][1]] = current_time

        else:
            if vessel_schedule[i-1][0] == 'PICK_UP':
                loading_time = vessel.get_loading_time(vessel_schedule[i-1][1].cargo_type, vessel_schedule[i-1][1].amount)
                if vessel_schedule[i][0] == 'DROP_OFF':
                    travel_distance = headquarters.get_network_distance(
                        vessel_schedule[i-1][1].origin_port,
                        vessel_schedule[i][1].destination_port)
                    travel_time = vessel.get_travel_time(travel_distance)
                    current_time += travel_time + loading_time

                    if current_time > latest_drop_off_time:  # later than the latest drop off time of next trade
                        return float('inf'), idle_time, pick_up_time, drop_off_time
                    if current_time < earliest_drop_off_time:  # earlier than the earliest drop off time of next trade
                        idle_time += earliest_drop_off_time - current_time
                        current_time = earliest_drop_off_time # update the current time
                    drop_off_time[vessel_schedule[i][1]] = current_time
                    #check if the last movement
                    if i == len(vessel_schedule) - 1:
                        current_time += loading_time # unloading time
                        loading_cost = vessel.get_unloading_consumption(loading_time) # unloading cost
                        cost += loading_cost
                        end_time = start_time + 720
                        idle_time += end_time - current_time

                elif vessel_schedule[i][0] == 'PICK_UP':
                    travel_distance = headquarters.get_network_distance(
                        vessel_schedule[i-1][1].origin_port,
                        vessel_schedule[i][1].origin_port)
                    travel_time = vessel.get_travel_time(travel_distance)
                    current_time += travel_time + loading_time
                    if current_time > latest_pick_up_time:  # later than the latest pick up time of next trade
                        return float('inf'), idle_time, pick_up_time, drop_off_time
                    if current_time < earliest_pick_up_time:  # earlier than the earliest pick up time of next trade
                        idle_time += earliest_pick_up_time - current_time
                        current_time = earliest_pick_up_time # update the current time
                    pick_up_time[vessel_schedule[i][1]] = current_time

                travel_laden_cost = vessel.get_laden_consumption(travel_time, vessel.speed)
                loading_cost = vessel.get_loading_consumption(loading_time)
                cost += travel_laden_cost + loading_cost
                current_hold_cargo += vessel_schedule[i-1][1].amount # add the cargo to the vessel

            elif vessel_schedule[i-1][0] == 'DROP_OFF':
                unloading_time = vessel.get_loading_time(
                    vessel_schedule[i-1][1].cargo_type,
                    vessel_schedule[i-1][1].amount)
                current_hold_cargo -= vessel_schedule[i-1][1].amount # remove the cargo from the vessel
                if vessel_schedule[i][0] == 'PICK_UP':
                    travel_distance = headquarters.get_network_distance(
                        vessel_schedule[i-1][1].destination_port,
                        vessel_schedule[i][1].origin_port)
                    travel_time = vessel.get_travel_time(travel_distance)
                    current_time += travel_time + unloading_time

                    if current_time > latest_pick_up_time:  # the latest pick up time
                        return float('inf'), idle_time, pick_up_time, drop_off_time

                    if current_time < earliest_pick_up_time:  # earlier than the earliest pick up time
                        idle_time += earliest_pick_up_time - current_time
                        current_time = earliest_pick_up_time # update the current time

                    pick_up_time[vessel_schedule[i][1]] = current_time

                elif vessel_schedule[i][0] == 'DROP_OFF':
                    travel_distance = headquarters.get_network_distance(
                        vessel_schedule[i-1][1].destination_port,
                        vessel_schedule[i][1].destination_port)

                    travel_time = vessel.get_travel_time(travel_distance)
                    current_time += travel_time + unloading_time

                    if current_time > latest_drop_off_time:  # later than the latest drop off time of next trade
                        return float('inf'), idle_time, pick_up_time, drop_off_time

                    if current_time < earliest_drop_off_time:  # earlier than the earliest drop off time of next trade
                        idle_time += earliest_drop_off_time - current_time
                        current_time = earliest_drop_off_time # update the current time

                    drop_off_time[vessel_schedule[i][1]] = current_time
                    #check if the last movement
                    if i == len(vessel_schedule) - 1:
                        current_time += unloading_time
                        unloading_cost = vessel.get_unloading_consumption(unloading_time) # unloading cost
                        cost += unloading_cost
                        end_time = start_time + 720
                        idle_time += end_time - current_time

                # check if the vessel is holding cargo
                if current_hold_cargo == 0 and vessel_schedule[i][0] == 'PICK_UP':
                    travel_ballast_cost = vessel.get_ballast_consumption(travel_time, vessel.speed)
                    cost += travel_ballast_cost
                else:
                    travel_laden_cost = vessel.get_laden_consumption(travel_time, vessel.speed)
                    cost += travel_laden_cost
                unloading_cost = vessel.get_unloading_consumption(unloading_time)
                cost += unloading_cost

    cost += vessel.get_idle_consumption(idle_time)
    if payments is not None:
        for trade in vessel_trades:
            cost -= payments[trade]

    return cost, idle_time, pick_up_time, drop_off_time

def cal_efficiency(schedules, headquarters, start_time):
    # calculate the total efficiency of the schedules
    actual_costs = 0
    absolute_costs = 1e-6  # prevent division by zero
    efficiency = 0
    for vessel, schedule in schedules.items():
        # get the actual cost of the schedule
        _, trades_specific_costs, _, _, _ = simulate_schedule_cost_allocated_shared_arrival(
            vessel,
            schedule,
            start_time,
            headquarters)
        for trade in schedule.get_scheduled_trades():
            travel_distance = headquarters.get_network_distance(trade.origin_port, trade.destination_port)
            travel_time = vessel.get_travel_time(travel_distance)
            travel_cost = vessel.get_laden_consumption(travel_time, vessel.speed)
            loading_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
            loading_cost = vessel.get_loading_consumption(loading_time)
            unloading_cost = vessel.get_unloading_consumption(loading_time)
            absolute_cost = travel_cost + loading_cost + unloading_cost
            if trade not in trades_specific_costs:
                raise ValueError(f"Trade {trade.origin_port} {trade.destination_port} not found in trades_specific_costs")
            actual_cost = trades_specific_costs[trade]
            actual_costs += actual_cost
            absolute_costs += absolute_cost

    efficiency = absolute_costs/actual_costs
    return efficiency

class Company7(TradingCompany):
    def __init__(self, fleet, name, profit_factor=1.4, profit_factor_2=1.2,
                 avg_w=0.7, cal_efficiency=True, schedule_with_greedy=False,
                 efficiency_selection_percentage=0.8, trade_frequency_threshold=0.5,
                 k_best=110, runtime_limit=57, pruning_factor=1.2):
        super().__init__(fleet, name)
        # --- hyper-parameters ---
        self._profit_factor = profit_factor
        self._profit_factor_2 = profit_factor_2
        self.avg_w = avg_w
        self.cal_efficiency = cal_efficiency
        self.schedule_with_greedy = schedule_with_greedy
        self.efficiency_selection_percentage = efficiency_selection_percentage
        self.trade_frequency_threshold = trade_frequency_threshold
        self.k_best = k_best
        self.runtime_limit = runtime_limit
        self.pruning_factor = pruning_factor
        # --- end of hyper-parameters ---
        self.total_cost_until_now = 0
        self.total_idle_time = 0

    @attrs.define
    class Data(TradingCompany.Data):
        profit_factor: float = 1.4
        profit_factor_2: float = 1.2
        avg_w: float = 0.7
        cal_efficiency: bool = True
        schedule_with_greedy: bool = False
        efficiency_selection_percentage: float = 0.8
        trade_frequency_threshold: float = 0.5
        k_best: int = 110
        runtime_limit: int = 57
        pruning_factor: float = 1.2
        class Schema(TradingCompany.Data.Schema):
            profit_factor = fields.Float(default=1.4)
            profit_factor_2 = fields.Float(default=1.2)
            avg_w = fields.Float(default=0.7)
            cal_efficiency = fields.Boolean(default=True)
            schedule_with_greedy = fields.Boolean(default=False)
            efficiency_selection_percentage = fields.Float(default=0.8)
            trade_frequency_threshold = fields.Float(default=0.5)
            k_best = fields.Integer(default=110)
            runtime_limit = fields.Integer(default=57)
            pruning_factor = fields.Float(default=1.2)
        # class Schema(TradingCompany.Data.Schema):
        #     profit_factor = fields.Float(default=1.65)


    def kbest_schedule(self, trades, fleets, headquarters, start_execution_time, num_current_solutions, payment_per_trade=None):
        # Add timer to track execution time

        best_vessel = None
        best_insertion_pickup_index = None
        best_insertion_dropoff_index = None
        start_time = trades[0].time
        # This dictionary holds the schedules *being built* during this specific function call only.
        schedules = {}

        for t, trade in enumerate(trades):
            # Check if time limit is about to be exceeded
            if time.time() - start_execution_time > self.runtime_limit:
                print(f"Time limit reached after processing {t}/{len(trades)} trades")
                break

            min_cost_for_all_vessels = float('inf')
            current_best_vessel = None
            current_best_insertion_pickup = None
            current_best_insertion_dropoff = None

            # Define a relative pruning factor (e.g., 1.5 means stop if vessel cost is 50% worse than the best found so far)
            # This could be made a class parameter if needed
            PRUNING_FACTOR = self.pruning_factor

            for v, vessel in enumerate(fleets):
                # Check time again for nested loop

                current_vessel_schedule = schedules.get(vessel, vessel.schedule)
                new_schedule_vessel = current_vessel_schedule.copy()
                insertion_points = new_schedule_vessel.get_insertion_points()

                min_cost_for_vessel = float('inf')
                vessel_best_insertion_pick_up = None
                vessel_best_insertion_drop_off = None
                stop_searching_this_vessel = False # Flag to break outer loop

                # Calculate the pruning threshold based on the best cost found in *previous* vessels
                # Only prune if a cost has actually been found (min_cost_for_all_vessels is not inf)
                pruning_threshold = min_cost_for_all_vessels * PRUNING_FACTOR if min_cost_for_all_vessels != float('inf') else float('inf')

                for i in range(1, len(insertion_points)+1):
                    for j in range(i, len(insertion_points)+1):
                        # Check time in the innermost loop
                        if time.time() - start_execution_time > self.runtime_limit:
                            stop_searching_this_vessel = True # Mark to break outer loops
                            break
                        try:
                            new_schedule_vessel_insertion = new_schedule_vessel.copy()
                            new_schedule_vessel_insertion.add_transportation(trade, i, j)
                        except Exception as e:
                            print(f"company {self.__class__.__name__} Error insert: {e}")
                            continue

                        if new_schedule_vessel_insertion.verify_schedule():
                            if len(new_schedule_vessel_insertion.get_simple_schedule()) % 2 != 0:
                                continue
                            try:
                                current_cost, _, _, _, _ = simulate_schedule_cost_allocated_shared_arrival(
                                    vessel,
                                    new_schedule_vessel_insertion,
                                    start_time,
                                    headquarters,
                                    payment_per_trade
                                )
                            except Exception as e:
                                print(f"company {self.__class__.__name__} Error simulate schedule cost: {e}")
                                continue
                            if current_cost < min_cost_for_vessel:
                                min_cost_for_vessel = current_cost
                                vessel_best_insertion_pick_up = i
                                vessel_best_insertion_drop_off = j

                                # *** Pruning Check ***
                                # If the best cost for *this* vessel is already worse than the threshold,
                                # stop searching further insertion points for this vessel.
                                if min_cost_for_vessel > pruning_threshold and num_current_solutions >= 5:
                                    stop_searching_this_vessel = True
                                    break # break j loop
                    # Break i loop if flagged by time limit or pruning check
                    if stop_searching_this_vessel or time.time() - start_execution_time > self.runtime_limit:
                        break

                # Break vessel loop if time limit reached
                if time.time() - start_execution_time > self.runtime_limit:
                    break

                # Update the overall best cost if this vessel provided a better one
                if min_cost_for_vessel < min_cost_for_all_vessels:
                    min_cost_for_all_vessels = min_cost_for_vessel
                    current_best_vessel = vessel
                    current_best_insertion_pickup = vessel_best_insertion_pick_up
                    current_best_insertion_dropoff = vessel_best_insertion_drop_off

            if current_best_vessel is not None:
                best_vessel = current_best_vessel
                best_insertion_pickup_index = current_best_insertion_pickup
                best_insertion_dropoff_index = current_best_insertion_dropoff
                best_vessel_schedule = schedules.get(best_vessel, best_vessel.schedule).copy()
                best_vessel_schedule.add_transportation(trade, best_insertion_pickup_index, best_insertion_dropoff_index)
                schedules[best_vessel] = best_vessel_schedule

        # Final check of execution time
        execution_time = time.time() - start_execution_time
        if execution_time > self.runtime_limit:
            print(f"Warning: kbest_schedule exceeded time limit: {execution_time:.2f} seconds")

        return schedules

    def propose_schedules(self, trades):
        costs = {}
        scheduled_trades = []
        rejected_trades = []
        rejection_threshold = 1000000
        pick_up_time = {}
        drop_off_time = {}
        start_time = trades[0].time
        start_execution_time = time.time()
        k_best_schedules = []
        k_best_schedule_costs = []
        kbest = self.k_best
        # shuffle the trades and generate kbest schedules
        for k in range(kbest):
            random.shuffle(trades)
            # schedules = {}
            schedule = self.kbest_schedule(trades, self._fleet, self._headquarters, start_execution_time, len(k_best_schedules))
            # record the cost of the schedule
            if len(schedule) > 0:
                k_best_schedules.append(schedule)

            time_end = time.time()
            if time_end - start_execution_time > self.runtime_limit:
                print(f"Time limit reached after generating {k}/{kbest} schedules")
                break
        time_end = time.time()
        print(f"Time taken: {time_end - start_execution_time} seconds")

        # ----- optional: calculate the efficiency of the k best schedules and sort them in descending order -----
        # calculate the efficiency of the k best schedules and sort them in descending order
        if self.cal_efficiency:
            k_efficiency = []
            for k_schedule in k_best_schedules:
                efficiency = cal_efficiency(k_schedule, self._headquarters, start_time)
                k_efficiency.append(efficiency)
            k_best_schedules = [x for _, x in sorted(zip(k_efficiency, k_best_schedules), key=lambda pair: pair[0], reverse=True)]

            # select the first 80% of the schedules according to the efficiency
            k_best_schedules = k_best_schedules[:int(len(k_best_schedules)*self.efficiency_selection_percentage)]
        # ----- end of optional -----

        # bid based on the average cost of the k best schedules
        if len(k_best_schedules) != 0:
            trade_frequencies, trade_avg_costs, rejected_trades = self.calculate_trade_frequency_and_avg_cost(
                k_best_schedules,
                len(k_best_schedules),
                self.trade_frequency_threshold,
                start_time)

            for trade, avg_cost in trade_avg_costs.items():
                # estimate the absolute cost of the trade OD
                travel_distance = self._headquarters.get_network_distance(trade.origin_port, trade.destination_port)
                travel_time = self._fleet[0].get_travel_time(travel_distance)
                travel_cost = self._fleet[0].get_laden_consumption(travel_time, self._fleet[0].speed)
                loading_time = self._fleet[0].get_loading_time(trade.cargo_type, trade.amount)
                loading_cost = self._fleet[0].get_loading_consumption(loading_time)
                unloading_cost = self._fleet[0].get_unloading_consumption(loading_time)
                absolute_cost = loading_cost + unloading_cost + travel_cost
                bid_price = self.avg_w * avg_cost + (1 - self.avg_w) * absolute_cost
                if bid_price < absolute_cost:
                    # costs[trade] = bid_price * self._profit_factor
                    costs[trade] = bid_price * ((absolute_cost - bid_price)/absolute_cost * (1.65 - self._profit_factor) + self._profit_factor)
                else:
                    costs[trade] = bid_price * self._profit_factor_2
                scheduled_trades.append(trade)

            # for the trades that are not scheduled, bid with high profit factor
            for trade in rejected_trades:
                # calculate the absolute cost of the trade OD
                travel_distance = self._headquarters.get_network_distance(trade.origin_port, trade.destination_port)
                travel_time = self._fleet[0].get_travel_time(travel_distance)
                travel_cost = self._fleet[0].get_laden_consumption(travel_time, self._fleet[0].speed)
                loading_time = self._fleet[0].get_loading_time(trade.cargo_type, trade.amount)
                loading_cost = self._fleet[0].get_loading_consumption(loading_time)
                unloading_cost = self._fleet[0].get_unloading_consumption(loading_time)
                absolute_cost = loading_cost + unloading_cost + travel_cost
                costs[trade] = absolute_cost * 10
                scheduled_trades.append(trade)

        # return ScheduleProposal(schedules, scheduled_trades, costs)
        return ScheduleProposal({}, scheduled_trades, costs)

    def receive(self, contracts, auction_ledger=None, *args, **kwargs):
        trades = [one_contract.trade for one_contract in contracts]
        payment_per_trade = {}
        for one_contract in contracts:
            payment_per_trade[one_contract.trade] = one_contract.payment

        scheduling_proposal = self.schedule_trades(trades, payment_per_trade)
        # Apply the schedules using the KBestBidComanyn's own apply_schedules method
        _ = self.apply_schedules(scheduling_proposal.schedules)

    def calculate_trade_frequency_and_avg_cost(self, k_best_schedules, kbest, frequency_threshold, start_time):
        # Dictionary to track which schedules each trade appears in
        trade_appearances = {}
        # Dictionary to track the total cost for each trade across all appearances
        trade_total_costs = {}
        # rejected trades
        rejected_trades = []
        # For each k-best schedule
        for k in range(kbest):
            # Track all trades in this schedule and their costs
            trades_in_schedule = set()
            trade_costs_in_schedule = {}

            total_cost = 0
            for vessel, schedule in k_best_schedules[k].items():
                # Get all trades scheduled on this vessel
                scheduled_trades = schedule.get_scheduled_trades()
                # Add to the set of trades in this schedule
                trades_in_schedule.update(scheduled_trades)
                # Calculate costs for these trades
                try:
                    trip_cost, trade_specific_costs, _, _, _ = simulate_schedule_cost_allocated_shared_arrival(
                        vessel,
                        schedule,
                        start_time,
                        self._headquarters)
                except Exception as e:
                    print(f"Error calculate_trade_frequency_and_avg_cost: {e}")
                    continue
                total_cost += trip_cost
                # record the cost for each trade
                for trade, cost in trade_specific_costs.items():
                    trade_costs_in_schedule[trade] = cost

            # Now update the appearance count and costs for all trades in this schedule
            for trade in trades_in_schedule:
                if trade not in trade_appearances:
                    trade_appearances[trade] = 0
                    trade_total_costs[trade] = 0

                trade_appearances[trade] += 1
                trade_total_costs[trade] += trade_costs_in_schedule.get(trade, 0)

        # Calculate average costs for trades that exceed the frequency threshold
        trade_avg_costs = {}
        trade_frequencies = {}

        for trade, appearances in trade_appearances.items():
            trade_frequencies[trade] = appearances / kbest  # Normalize to get frequency between 0 and 1

            if trade_frequencies[trade] >= frequency_threshold:
                # Calculate average cost for this trade
                trade_avg_costs[trade] = trade_total_costs[trade] / appearances
            else:
                rejected_trades.append(trade)

        return trade_frequencies, trade_avg_costs, rejected_trades


    def schedule_trades(self, trades, payment_per_trade):
        scheduled_trades = []
        schedules = {}
        costs = {}
        if len(trades) == 0:
            return ScheduleProposal(schedules, scheduled_trades, costs)
        k_best_schedules = []
        kbest = self.k_best
        start_time = trades[0].time
        start_execution_time = time.time()
        for k in range(kbest):
            random.shuffle(trades)
            # schedules = {}
            schedule = self.kbest_schedule(trades, self._fleet, self._headquarters, start_execution_time, len(k_best_schedules), payment_per_trade)
            if len(schedule) > 0:
                k_best_schedules.append(schedule)
            end_time = time.time()
            if end_time - start_execution_time > self.runtime_limit:
                print(f"In schedule_trades: Time limit reached after generating {k}/{kbest} schedules")
                break

        # choose the minimum cost schedule
        min_cost = float('inf')
        min_cost_schedule_index = -1
        for k, k_schedule in enumerate(k_best_schedules):
            schedule_total_cost = 0
            for vessel, schedule in k_schedule.items():
                cost, _, _, _ = simulate_schedule_cost(
                    vessel,
                    schedule,
                    start_time,
                    self._headquarters,
                    payment_per_trade)
                schedule_total_cost += cost

            if schedule_total_cost < min_cost:
                min_cost = schedule_total_cost
                min_cost_schedule_index = k

        if min_cost_schedule_index >= 0:
            schedules = k_best_schedules[min_cost_schedule_index]

        return ScheduleProposal(schedules, scheduled_trades, costs)


