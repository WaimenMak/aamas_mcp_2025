# -*- coding: utf-8 -*-
# @Time    : 23/04/2025 15:54
# @Author  : mmai
# @FileName: kbest
# @Software: PyCharm
from turtledemo.penrose import start

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
import random
from greedy import simulate_schedule_cost

class KBestComanyn(TradingCompany):
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


    def kbest_schedule(self, trades, fleets, schedules, headquarters):

        # min_cost_for_trades = float('inf')
        # best_trade = None
        best_vessel = None
        # best_pickup_time = None
        # best_dropoff_time = None
        best_insertion_pickup_index = None
        best_insertion_dropoff_index = None
        start_time = trades[0].time

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
                current_vessel_schedule = schedules.get(vessel, vessel.schedule)
                new_schedule_vessel = current_vessel_schedule.copy()
                insertion_points = new_schedule_vessel.get_insertion_points()

                min_cost_for_vessel = float('inf')
                vessel_best_insertion_pick_up = None
                vessel_best_insertion_drop_off = None
                # vessel_best_pickup = None
                # vessel_best_dropoff = None

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
                                new_schedule_vessel_insertion.get_simple_schedule(),
                                start_time,
                                headquarters
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
                best_vessel = current_best_vessel
                best_insertion_pickup_index = current_best_insertion_pickup
                best_insertion_dropoff_index = current_best_insertion_dropoff
                best_vessel_schedule = schedules.get(best_vessel, best_vessel.schedule)
                best_vessel_schedule.add_transportation(trade, best_insertion_pickup_index, best_insertion_dropoff_index)
                schedules[best_vessel] = best_vessel_schedule


        return schedules
            # No feasible assignment found
            # return float('inf'), None, None, None, None, None

    def propose_schedules(self, trades):

        costs = {}
        scheduled_trades = []
        rejection_threshold = 1000000
        rejected_trades = []
        pick_up_time = {}
        drop_off_time = {}
        start_time = trades[0].time
        time_start = time.time()
        k_best_schedules = []
        kbest = 50
        # shuffle the trades and generate kbest schedules
        for k in range(kbest):
            random.shuffle(trades)
            schedules = {}
            k_best_schedules.append(self.kbest_schedule(trades, self._fleet, schedules, self._headquarters))
            time_end = time.time()
            # if time_end - time_start > 3:
            #     break
        # print(f"Time taken: {time_end - time_start} seconds")

        # First, find the minimum cost schedule
        min_cost = float('inf')
        min_cost_schedule_index = -1

        for k, k_schedule in enumerate(k_best_schedules):
            schedule_total_cost = 0
            # for vessel, schedule in k_schedule.items():
            #     cost, idle_time, pickup, dropoff = simulate_schedule_cost(vessel, schedule.get_simple_schedule(), start_time, self._headquarters)
            #     schedule_total_cost += cost
            for vessel in self._fleet:
                if vessel in k_schedule:
                    schedule = k_schedule[vessel]
                    cost, idle_time, pickup, dropoff = simulate_schedule_cost(
                        vessel,
                        schedule.get_simple_schedule(),
                        start_time,
                        self._headquarters)
                else:
                    cost, idle_time, pickup, dropoff = simulate_schedule_cost(
                        vessel,
                        [],
                        start_time,
                        self._headquarters)
                schedule_total_cost += cost
            # Track the minimum cost schedule
            if schedule_total_cost < min_cost:
                min_cost = schedule_total_cost
                min_cost_schedule_index = k

        # Now calculate costs only for trades in the minimum cost schedule
        if min_cost_schedule_index >= 0:  # Ensure we found a valid schedule
            min_cost_schedule = k_best_schedules[min_cost_schedule_index]
            
            for vessel, schedule in min_cost_schedule.items():
                for trade in schedule.get_scheduled_trades():
                    loading_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
                    unloading_cost = vessel.get_unloading_consumption(loading_time)
                    loading_cost = vessel.get_loading_consumption(loading_time)
                    travel_distance = self._headquarters.get_network_distance(trade.origin_port, trade.destination_port)
                    travel_time = vessel.get_travel_time(travel_distance)
                    travel_cost = vessel.get_laden_consumption(travel_time, vessel.speed)
                    trade_cost = loading_cost + unloading_cost + travel_cost
                    costs[trade] = trade_cost * self._profit_factor
                    scheduled_trades.append(trade)

        print(f"Minimum schedule cost: {min_cost}")
        print(f"Number of trades in minimum cost schedule: {len(costs)}")


        return ScheduleProposal(schedules, scheduled_trades, costs)

    def schedule_trades(self, trades):
        scheduled_trades = []
        schedules = {}
        costs = {}
        if len(trades) == 0:
            return ScheduleProposal(schedules, scheduled_trades, costs)
        k_best_schedules = []
        kbest = 50
        start_time = trades[0].time
        for k in range(kbest):
            random.shuffle(trades)
            schedules = {}
            k_best_schedules.append(self.kbest_schedule(trades, self._fleet, schedules, self._headquarters))
            
        # choose the minimum cost schedule
        min_cost = float('inf')
        min_cost_schedule_index = -1
        for k, k_schedule in enumerate(k_best_schedules):
            schedule_total_cost = 0
            for vessel, schedule in k_schedule.items():
                cost, idle_time, pickup, dropoff = simulate_schedule_cost(vessel, schedule.get_simple_schedule(), start_time, self._headquarters)
                schedule_total_cost += cost
            
            if schedule_total_cost < min_cost:
                min_cost = schedule_total_cost
                min_cost_schedule_index = k

        schedules = k_best_schedules[min_cost_schedule_index]
        # for vessel, schedule in schedules.items():
        #     for trade in schedule.get_scheduled_trades():
        #         scheduled_trades.append(trade)
        #         loading_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
        #         unloading_cost = vessel.get_unloading_consumption(loading_time)
        #         loading_cost = vessel.get_loading_consumption(loading_time)
        #         travel_distance = self._headquarters.get_network_distance(trade.origin_port, trade.destination_port)
        #         travel_time = vessel.get_travel_time(travel_distance)
        #         travel_cost = vessel.get_laden_consumption(travel_time, vessel.speed)
        #         trade_cost = loading_cost + unloading_cost + travel_cost
                # costs[trade] = 0
                
        return ScheduleProposal(schedules, scheduled_trades, costs)

    def receive(self, contracts, auction_ledger=None, *args, **kwargs):
        trades = [one_contract.trade for one_contract in contracts]
        # scheduling_proposal = self.propose_schedules(trades)
        scheduling_proposal = self.schedule_trades(trades)
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
                cost, idle_time, pickup, dropoff = simulate_schedule_cost(vessel, schedule.get_simple_schedule(), start_time, self._headquarters)
                total_cost += cost
                # Distribute the cost among the trades on this vessel
                # This could be done proportionally based on trade characteristics
                # For simplicity, we'll divide equally here
                if scheduled_trades:
                    cost_per_trade = cost / len(scheduled_trades)
                    for trade in scheduled_trades:
                        trade_costs_in_schedule[trade] = cost_per_trade
            
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





