# -*- coding: utf-8 -*-
# @Time    : 15/04/2025 22:22
# @Author  : mmai
# @FileName: greedy
# @Software: PyCharm

import numpy as np
from mable.cargo_bidding import TradingCompany, SimpleCompany
from mable.examples.companies import ScheduleProposal
import attrs
from marshmallow import fields
import time
from utils import simulate_schedule_cost_allocated_shared_arrival


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

class GreedyComanyn(TradingCompany):
    def __init__(self, fleet, name, profit_factor=1.65):
        super().__init__(fleet, name)
        self._profit_factor = profit_factor
        self.total_cost_until_now = 0
        self.total_idle_time = 0

    @attrs.define
    class Data(TradingCompany.Data):
        profit_factor: float = 1.65

        class Schema(TradingCompany.Data.Schema):
            profit_factor = fields.Float(default=1.65)


    def greedy_schedule(self, trades, fleets, schedules, scheduled_trades, headquarters, payments=None):

        min_cost_for_trades = float('inf')
        best_trade = None
        best_vessel = None
        best_pickup_time = None
        best_dropoff_time = None
        best_insertion_pickup_index = None
        best_insertion_dropoff_index = None
        start_time = trades[0].time

        for t, trade in enumerate(trades):
            if trade in scheduled_trades:
                continue
            min_cost_for_all_vessels = float('inf')
            current_best_vessel = None
            current_best_pickup = None
            current_best_dropoff = None
            current_best_insertion_pickup = None
            current_best_insertion_dropoff = None
            
            for v, vessel in enumerate(fleets):
                current_vessel_schedule = schedules.get(vessel, vessel.schedule)
                new_schedule_vessel = current_vessel_schedule.copy()
                insertion_points = new_schedule_vessel.get_insertion_points()

                min_cost_for_vessel = float('inf')
                vessel_best_insertion_pick_up = None
                vessel_best_insertion_drop_off = None
                vessel_best_pickup = None
                vessel_best_dropoff = None
                
                for i in range(1, len(insertion_points)+1):
                    if len(insertion_points) > 1:
                        pass
                    for j in range(i, len(insertion_points)+1):
                        new_schedule_vessel_insertion = new_schedule_vessel.copy()
                        # try to add trade to vessel schedule with all possible insertion points
                        new_schedule_vessel_insertion.add_transportation(trade, i, j)
                            
                        # if new_schedule_vessel_insertion.verify_schedule_cargo():
                        if new_schedule_vessel_insertion.verify_schedule():
                            current_cost, idle_time, pickup, dropoff = simulate_schedule_cost(
                                vessel, 
                                new_schedule_vessel_insertion,
                                start_time,
                                headquarters,
                                payments
                            )
                            if current_cost < min_cost_for_vessel:
                                min_cost_for_vessel = current_cost
                                vessel_best_insertion_pick_up = i
                                vessel_best_insertion_drop_off = j
                                vessel_best_pickup = pickup
                                vessel_best_dropoff = dropoff

                if min_cost_for_vessel < min_cost_for_all_vessels:
                    min_cost_for_all_vessels = min_cost_for_vessel
                    current_best_vessel = vessel
                    current_best_pickup = vessel_best_pickup
                    current_best_dropoff = vessel_best_dropoff
                    current_best_insertion_pickup = vessel_best_insertion_pick_up
                    current_best_insertion_dropoff = vessel_best_insertion_drop_off

            if min_cost_for_all_vessels < min_cost_for_trades:
                min_cost_for_trades = min_cost_for_all_vessels
                best_trade = trade
                best_vessel = current_best_vessel
                best_pickup_time = current_best_pickup
                best_dropoff_time = current_best_dropoff
                best_insertion_pickup_index = current_best_insertion_pickup
                best_insertion_dropoff_index = current_best_insertion_dropoff

        # Check if we found a feasible assignment
        if best_trade is not None and best_vessel is not None:
            # Optional: Update the vessel's schedule with the best trade found
            best_vessel_schedule = schedules.get(best_vessel, best_vessel.schedule)
            best_vessel_schedule.add_transportation(best_trade, best_insertion_pickup_index, best_insertion_dropoff_index)
            # schedules[best_vessel] = best_vessel.schedule
            # schedules[best_vessel] = best_vessel_schedule
            # scheduled_trades.append(best_trade)
            
            # Calculate the cost components using the saved best values
            load_time_best_trade = best_vessel.get_loading_time(best_trade.cargo_type, best_trade.amount)
            loading_cost = best_vessel.get_loading_consumption(load_time_best_trade)
            unloading_cost = best_vessel.get_unloading_consumption(load_time_best_trade)
            
            # Use the best_pickup_time and best_dropoff_time that correspond to the best assignment
            travel_time = best_dropoff_time[best_trade] - best_pickup_time[best_trade]
            # travel_time = best_vessel.get_travel_time(headquarters.get_network_distance(best_trade.origin_port, best_trade.destination_port))
            travel_cost = best_vessel.get_laden_consumption(travel_time, best_vessel.speed)
            total_cost = loading_cost + unloading_cost + travel_cost
            
            return total_cost, best_trade, best_vessel, best_vessel_schedule, best_pickup_time, best_dropoff_time
        else:
            # No feasible assignment found
            return float('inf'), None, None, None, None, None
    
    def propose_schedules(self, trades, payment_per_trade=None):
        # for v, vessel in enumerate(self._fleet):
        #     if len(vessel.schedule.get_simple_schedule()) == 1:
        #         pass
        schedules = {}
        costs = {}
        scheduled_trades = []
        if len(trades) == 0:
            return ScheduleProposal(schedules, scheduled_trades, costs)
        rejection_threshold = 1000000
        last_rejected_trade = None
        rejected_trades = []
        current_trade = None
        pick_up_time = {}
        drop_off_time = {}
        start_time = trades[0].time
        time_start = time.time()
        while len(scheduled_trades) < len(trades):
            # if len(rejected_trades) > 1:
            #     pass
            for trade in trades:
                current_trade = trade
                if trade not in scheduled_trades:
                    cost_trade, trade, best_vessel, best_vessel_schedule, best_pickup_time, best_dropoff_time = self.greedy_schedule(
                        trades, 
                        self._fleet, 
                        schedules, 
                        scheduled_trades, 
                        self._headquarters,
                        payment_per_trade
                    )
                    if cost_trade > rejection_threshold:
                        last_rejected_trade = current_trade
                        rejected_trades.append(current_trade)
                        continue
                    scheduled_trades.append(trade)
                    schedules[best_vessel] = best_vessel_schedule
                    pick_up_time[trade] = best_pickup_time[trade]
                    drop_off_time[trade] = best_dropoff_time[trade]
                    # costs[trade] = cost_trade * self._profit_factor  # naive calculate the cost based on travel time
            time_end = time.time()
            if time_end - time_start > 3 or last_rejected_trade == current_trade:
            # if last_rejected_trade == current_trade:
                break
        # print(f"Time taken: {time_end - time_start} seconds")

        #simulate cost with connection cost and accurately calculate the shared cost
        for vessel, schedule in schedules.items():
            trip_cost, trade_specific_costs, _, _, _ = simulate_schedule_cost_allocated_shared_arrival(vessel, schedule, start_time, self._headquarters)
            for trade in schedule.get_scheduled_trades():
                # calculate absolute cost
                travel_distance = self._headquarters.get_network_distance(trade.origin_port, trade.destination_port)
                travel_time = vessel.get_travel_time(travel_distance)
                travel_cost = vessel.get_laden_consumption(travel_time, vessel.speed)
                loading_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
                loading_cost = vessel.get_loading_consumption(loading_time)
                unloading_cost = vessel.get_unloading_consumption(loading_time)
                absolute_cost = loading_cost + unloading_cost + travel_cost
                # costs[trade] = trade_specific_costs[trade] * self._profit_factor
                if trade_specific_costs[trade] < absolute_cost:
                    costs[trade] = trade_specific_costs[trade] * self._profit_factor
                else:
                    costs[trade] = trade_specific_costs[trade] * 1.2
        
        # for vessel in self._fleet:
        #     if vessel in schedules:
        #         schedule = schedules[vessel]
        #         cost, idle_time, pickup, dropoff = simulate_schedule_cost_allocated_shared_arrival(
        #             vessel,
        #             schedule,
        #             start_time,
        #             self._headquarters)
        #     else:
        #         cost, idle_time, pickup, dropoff = simulate_schedule_cost(
        #             vessel,
        #             [],
        #             start_time,
        #             self._headquarters)

            # for trade in schedule.get_scheduled_trades():
            #     costs[trade] = cost/len(schedule.get_scheduled_trades()) * self._profit_factor

            # self.total_cost_until_now += cost
            # self.total_idle_time += idle_time

        # print(f"Total cost until now: {self.total_cost_until_now}")
        # print(f"Total idle time until now: {self.total_idle_time}")

        # split the cost by the number of trades on vessel schedule
        # for vessel, schedule in schedules.items():
        #     cost, idle_time, pickup, dropoff = self.simulate_shcedule_cost(vessel, schedule.get_simple_schedule(), self._headquarters)
        #     for trade in schedule.get_scheduled_trades():
        #         costs[trade] = cost/len(schedule.get_scheduled_trades()) * self._profit_factor

        # accurately calculate the cost based on the actual schedule, overlap trades divide the cost by the number of trades on vessel schedule
        # for vessel, schedule in schedules.items():
        #     trades_list = schedule.get_scheduled_trades() # already sorted by time
        #     for i in range(len(trades_list)):
        #         # check if overlap with the following trade
        #         for j in range(i+1, len(trades_list)):
        #             if drop_off_time[trades_list[i]] > pick_up_time[trades_list[j]]:
        #                 overlap_time = drop_off_time[trades_list[i]] - pick_up_time[trades_list[j]]
        #                 overlap_cost = vessel.get_laden_consumption(overlap_time, vessel.speed)
        #                 costs[trades_list[i]] -= overlap_cost / 2
        #                 costs[trades_list[j]] -= overlap_cost / 2
        #                 if costs[trades_list[i]] < 0:
        #                     print(f"Cost for trade {trades_list[i]} is negative: {costs[trades_list[i]]}")
        #                 if costs[trades_list[j]] < 0:
        #                     print(f"Cost for trade {trades_list[j]} is negative: {costs[trades_list[j]]}")


        return ScheduleProposal(schedules, scheduled_trades, costs)

    def receive(self, contracts, auction_ledger=None, *args, **kwargs):
        trades = [one_contract.trade for one_contract in contracts]
        payment_per_trade = {}
        for one_contract in contracts:
            payment_per_trade[one_contract.trade] = one_contract.payment
        scheduling_proposal = self.propose_schedules(trades, payment_per_trade)
        _ = self.apply_schedules(scheduling_proposal.schedules)


            

                        
                
