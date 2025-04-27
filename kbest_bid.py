# -*- coding: utf-8 -*-
# @Time    : 24/04/2025 23:19
# @Author  : mmai
# @FileName: kbest_bid
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time    : 23/04/2025 15:54
# @Author  : mmai
# @FileName: kbest
# @Software: PyCharm

from mable.cargo_bidding import TradingCompany, SimpleCompany
from mable.examples.companies import ScheduleProposal
import attrs
from marshmallow import fields
import time
import random
random.seed(1)
from collections import defaultdict
from greedy import GreedyComanyn # Added alias if needed

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
    try:
        # Add handling for None time_window during sorting
        def sort_key(item):
            event_type, trade = item
            if trade.time_window is None:
                # If no time window, treat as unbounded for sorting
                return (float('-inf'), float('inf'))
            if event_type == 'PICK_UP':
                return (trade.time_window[0], trade.time_window[1])
            else: # DROP_OFF
                return (trade.time_window[2], trade.time_window[3])

        vessel_schedule.sort(key=sort_key)
    except AttributeError:
        print("Warning: Could not sort vessel schedule based on time windows (AttributeError).")
        pass
    except IndexError:
        print("Warning: Could not sort vessel schedule based on time windows (IndexError - possibly incomplete time_window).")
        pass

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

def get_costs_for_schedule(schedule, fleets, headquarters, start_time):
    schedule_total_cost = 0
    for vessel in fleets:
        if vessel in schedule:
            schedule = schedule[vessel]
            cost, _, _, _, _ = simulate_schedule_cost_allocated_shared_arrival(
                vessel,
                schedule,
                start_time,
                headquarters)
        else:
            cost, _, _, _, _ = simulate_schedule_cost_allocated_shared_arrival(
                vessel,
                [],
                start_time,
                headquarters)
        schedule_total_cost += cost
    return schedule_total_cost

class KBestBidComanyn(TradingCompany):
    def __init__(self, fleet, name, profit_factor=1.65):
        super().__init__(fleet, name)
        self._profit_factor = profit_factor
        self.total_cost_until_now = 0
        self.total_idle_time = 0
        self.k_best = 100
        # random.seed(1)

    @attrs.define
    class Data(TradingCompany.Data):
        profit_factor: float = 1.65

        class Schema(TradingCompany.Data.Schema):
            profit_factor = fields.Float(default=1.65)


    def kbest_schedule(self, trades, fleets, headquarters, payment_per_trade=None):

        # min_cost_for_trades = float('inf')
        # best_trade = None
        best_vessel = None
        # best_pickup_time = None
        # best_dropoff_time = None
        best_insertion_pickup_index = None
        best_insertion_dropoff_index = None
        start_time = trades[0].time
        # This dictionary holds the schedules *being built* during this specific function call only.
        schedules = {}
        # for v, vessel in enumerate(fleets):
        #     if len(vessel.schedule.get_simple_schedule()) == 3:
        #         pass
        for t, trade in enumerate(trades):
            # if trade in scheduled_trades:
            #     continue
            min_cost_for_all_vessels = float('inf')
            current_best_vessel = None
            # current_best_pickup = None
            # current_best_dropoff = None
            current_best_insertion_pickup = None
            current_best_insertion_dropoff = None

            for v, vessel in enumerate(fleets):
                # if start_time == 720:
                #     pass
                # if len(vessel.schedule.get_simple_schedule()) == 1:
                #     pass
                current_vessel_schedule = schedules.get(vessel, vessel.schedule)
                new_schedule_vessel = current_vessel_schedule.copy()
                insertion_points = new_schedule_vessel.get_insertion_points()

                min_cost_for_vessel = float('inf')
                vessel_best_insertion_pick_up = None
                vessel_best_insertion_drop_off = None
                # vessel_best_pickup = None
                # vessel_best_dropoff = None

                for i in range(1, len(insertion_points)+1):
                    for j in range(i, len(insertion_points)+1):
                        new_schedule_vessel_insertion = new_schedule_vessel.copy()
                        # try to add trade to vessel schedule with all possible insertion points
                        # if len(new_schedule_vessel_insertion.get_simple_schedule()) == 1:
                        #     pass
                        new_schedule_vessel_insertion.add_transportation(trade, i, j)
                        # if len(new_schedule_vessel_insertion.get_simple_schedule()) == 3:
                        #     pass
                        # if new_schedule_vessel_insertion.verify_schedule_cargo():
                        if new_schedule_vessel_insertion.verify_schedule():
                            if len(new_schedule_vessel_insertion.get_simple_schedule()) % 2 != 0:
                                continue

                            # scheduled_trades = new_schedule_vessel_insertion.get_scheduled_trades()
                            current_cost, _, _, _, _ = simulate_schedule_cost_allocated_shared_arrival(
                                vessel,
                                new_schedule_vessel_insertion,
                                start_time,
                                headquarters,
                                payment_per_trade
                            )
                            if current_cost < min_cost_for_vessel:
                                min_cost_for_vessel = current_cost
                                vessel_best_insertion_pick_up = i
                                vessel_best_insertion_drop_off = j
                                # vessel_best_pickup = pickup
                                # vessel_best_dropoff = dropoff


                if min_cost_for_vessel < min_cost_for_all_vessels:
                    min_cost_for_all_vessels = min_cost_for_vessel
                    current_best_vessel = vessel
                    # current_best_pickup = vessel_best_pickup
                    # current_best_dropoff = vessel_best_dropoff
                    current_best_insertion_pickup = vessel_best_insertion_pick_up
                    current_best_insertion_dropoff = vessel_best_insertion_drop_off

            if current_best_vessel is not None:
                # if current_best_vessel == fleets[1] and start_time == 720:
                #     pass
                best_vessel = current_best_vessel
                best_insertion_pickup_index = current_best_insertion_pickup
                best_insertion_dropoff_index = current_best_insertion_dropoff
                best_vessel_schedule = schedules.get(best_vessel, best_vessel.schedule).copy()
                best_vessel_schedule.add_transportation(trade, best_insertion_pickup_index, best_insertion_dropoff_index)
                schedules[best_vessel] = best_vessel_schedule


        return schedules

    def propose_schedules(self, trades):
        # for v, vessel in enumerate(self._fleet):
        #     if len(vessel.schedule.get_simple_schedule()) ==3:
        #         pass
        costs = {}
        scheduled_trades = []
        rejection_threshold = 1000000
        rejected_trades = []
        pick_up_time = {}
        drop_off_time = {}
        start_time = trades[0].time
        time_start = time.time()
        k_best_schedules = []
        k_best_schedule_costs = []
        kbest = self.k_best
        # shuffle the trades and generate kbest schedules
        time_start = time.time()
        for k in range(kbest):
            random.shuffle(trades)
            # schedules = {}
            schedule = self.kbest_schedule(trades, self._fleet, self._headquarters)
            # record the cost of the schedule
            if len(schedule) > 0:
                k_best_schedules.append(schedule)
                schedule_cost = get_costs_for_schedule(schedule, self._fleet, self._headquarters, start_time)
                k_best_schedule_costs.append(schedule_cost)

            time_end = time.time()
            # if time_end - time_start > 3:
            #     break
        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")

        # get the minimum cost schedule
        if len(k_best_schedule_costs) != 0:
            min_cost_schedule_index = k_best_schedule_costs.index(min(k_best_schedule_costs))
            schedules = k_best_schedules[min_cost_schedule_index]
        # -- bid based on average cost of k best schedules
        if len(k_best_schedules) != 0:
            trade_frequencies, trade_avg_costs = self.calculate_trade_frequency_and_avg_cost(
                k_best_schedules,
                kbest,
                0.5,
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
                if avg_cost < absolute_cost:
                    costs[trade] = avg_cost * self._profit_factor
                else:
                    costs[trade] = avg_cost * 1
                scheduled_trades.append(trade)


        # return ScheduleProposal(schedules, scheduled_trades, costs)
        return ScheduleProposal({}, scheduled_trades, costs)

    def receive(self, contracts, auction_ledger=None, *args, **kwargs):
        trades = [one_contract.trade for one_contract in contracts]
        payment_per_trade = {}
        for one_contract in contracts:
            payment_per_trade[one_contract.trade] = one_contract.payment
        # scheduling_proposal = self.schedule_trades(trades, payment_per_trade)
        # --- Use GreedyComanyn's propose_schedules logic ---
        # 1. Create a temporary instance of GreedyComanyn using this company's fleet/hq/etc.
        #    (Assumes __init__ signatures are compatible or GreedyComanyn doesn't need specific state)
        temp_greedy_company = GreedyComanyn(self._fleet, self.name, self._profit_factor)
        # 2. Set up the headquarters for the temporary instance if needed (standard pattern)
        temp_greedy_company._headquarters = self._headquarters
        # 3. Call the propose_schedules method on the temporary instance
        scheduling_proposal = temp_greedy_company.propose_schedules(trades, payment_per_trade)
        # --- End of Greedy logic usage ---

        # Apply the schedules using the KBestBidComanyn's own apply_schedules method
        _ = self.apply_schedules(scheduling_proposal.schedules)

    def calculate_trade_frequency_and_avg_cost(self, k_best_schedules, kbest, frequency_threshold, start_time):
        # Dictionary to track which schedules each trade appears in
        trade_appearances = {}
        # Dictionary to track the total cost for each trade across all appearances
        trade_total_costs = {}
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
                trip_cost, trade_specific_costs, _, _, _ = simulate_schedule_cost_allocated_shared_arrival(
                    vessel,
                    schedule,
                    start_time,
                    self._headquarters)
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

        return trade_frequencies, trade_avg_costs


    def schedule_trades(self, trades, payment_per_trade):
        # for v, vessel in enumerate(self._fleet):
        #     if len(vessel.schedule) != 0:
        #         pass
        # if len(trades) == 0:
        #     pass
        scheduled_trades = []
        schedules = {}
        costs = {}
        if len(trades) == 0:
            return ScheduleProposal(schedules, scheduled_trades, costs)
        k_best_schedules = []
        kbest = self.k_best
        start_time = trades[0].time
        for k in range(kbest):
            random.shuffle(trades)
            # schedules = {}
            schedule = self.kbest_schedule(trades, self._fleet, self._headquarters, payment_per_trade)
            if len(schedule) > 0:
                k_best_schedules.append(schedule)

        # choose the minimum cost schedule
        min_cost = float('inf')
        min_cost_schedule_index = -1
        for k, k_schedule in enumerate(k_best_schedules):
            schedule_total_cost = 0
            for vessel, schedule in k_schedule.items():
                cost, _, _, _, _ = simulate_schedule_cost_allocated_shared_arrival(
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


