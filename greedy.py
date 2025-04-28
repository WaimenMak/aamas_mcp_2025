# -*- coding: utf-8 -*-
# @Time    : 15/04/2025 22:22
# @Author  : mmai
# @FileName: greedy
# @Software: PyCharm

from mable.cargo_bidding import TradingCompany
from mable.examples.companies import ScheduleProposal
import attrs
from marshmallow import fields
import time
from utils import simulate_schedule_cost_allocated_shared_arrival, simulate_schedule_cost

class GreedyComanyn(TradingCompany):
    def __init__(self, fleet, name, profit_factor=1.65, runtime_limit=55):
        super().__init__(fleet, name)
        self._profit_factor = profit_factor
        self.total_cost_until_now = 0
        self.total_idle_time = 0
        self.runtime_limit = runtime_limit
    @attrs.define
    class Data(TradingCompany.Data):
        profit_factor: float = 1.65
        runtime_limit: int = 55
        class Schema(TradingCompany.Data.Schema):
            profit_factor = fields.Float(default=1.65)
            runtime_limit = fields.Integer(default=55)

    def greedy_schedule(self, trades, fleets, schedules, scheduled_trades, headquarters, start_execution_time, payments=None):

        min_cost_for_trades = float('inf')
        best_trade = None
        best_vessel = None
        best_pickup_time = None
        best_dropoff_time = None
        best_insertion_pickup_index = None
        best_insertion_dropoff_index = None
        start_time = trades[0].time
    
        for t, trade in enumerate(trades):
            if time.time() - start_execution_time > self.runtime_limit:
                break
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
                    for j in range(i, len(insertion_points)+1):
                        if time.time() - start_execution_time > self.runtime_limit:
                            break
                        try:
                            new_schedule_vessel_insertion = new_schedule_vessel.copy()
                            # try to add trade to vessel schedule with all possible insertion points
                            new_schedule_vessel_insertion.add_transportation(trade, i, j)
                        except Exception as e:
                            print(f"Error insert: {e}")
                            continue
                        # if new_schedule_vessel_insertion.verify_schedule_cargo():
                        if new_schedule_vessel_insertion.verify_schedule():
                            if len(new_schedule_vessel_insertion.get_simple_schedule()) % 2 != 0:
                                continue
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
            best_vessel_schedule = schedules.get(best_vessel, best_vessel.schedule).copy()
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
        start_execution_time = time.time()
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
                        start_execution_time,
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
            if time_end - start_execution_time > self.runtime_limit or last_rejected_trade == current_trade:
            # if last_rejected_trade == current_trade:
                break
        # print(f"Time taken: {time_end - time_start} seconds")

        #simulate cost with connection cost and accurately calculate the shared cost
        for vessel, schedule in schedules.items():
            if schedule.verify_schedule():
                try:
                    trip_cost, trade_specific_costs, _, _, _ = simulate_schedule_cost_allocated_shared_arrival(vessel, schedule, start_time, self._headquarters)
                except Exception as e:
                    print(f"Error simulating schedule cost: {e}")
                    continue
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
                        costs[trade] = trade_specific_costs[trade] * 1.3
            return ScheduleProposal(schedules, scheduled_trades, costs)
        
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


            

                        
                
