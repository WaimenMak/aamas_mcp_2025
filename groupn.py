# -*- coding: utf-8 -*-
# @Time    : 09/04/2025 11:27
# @Author  : mmai
# @FileName: groupn
# @Software: PyCharm

from mable.cargo_bidding import TradingCompany, SimpleCompany
from mable.examples.companies import ScheduleProposal
from marshmallow import fields
import attrs
from ortools.sat.python import cp_model

from loguru import logger
from mable.cargo_bidding import Bid
from copy import deepcopy
from math import ceil
import numpy as np
from collections import defaultdict
from Agents import Solver




# class Companyn(TradingCompany):
#
#     def pre_inform(self, trades, time):
#         logger.warning("pre_inform")
#         _ = self.propose_schedules(trades)

    # def inform(self, trades):
    #     pass
    #
    # def propose_schedules(self, trades):
    #     schedules = {}
    #     scheduled_trades = []

class OurCompanyn(TradingCompany):
    def __init__(self, fleet, name, profit_factor=1.65):
        super().__init__(fleet, name)
        self._profit_factor = profit_factor

    @attrs.define
    class Data(TradingCompany.Data):
        profit_factor: float = 1.65

        class Schema(TradingCompany.Data.Schema):
            profit_factor = fields.Float(default=1.65)
    # def pre_inform(self, trades, time):
    #     logger.warning("pre_inform")
    #     pass
        # _ = self.propose_schedules(trades)

    # def inform(self, trades, *args, **kwargs):
    #
    #     self.propose_schedules(trades)
    #     # scheduled_trades = proposed_scheduling.scheduled_trades
    #     # trades_and_costs = [
    #     #     (x, -100)
    #     #     for x in scheduled_trades
    #     # ]
    #     #
    #     # bids = [Bid(amount=cost, trade=one_trade) for one_trade, cost in trades_and_costs]
    #     # return bids
    #
    # def receive(self, contracts, auction_ledger=None, *args, **kwargs):
    #     trades = [one_contract.trade for one_contract in contracts]
    #     scheduling_proposal = self.propose_schedules(trades)
    #     rejected_trades = self.apply_schedules(scheduling_proposal.schedules)
    #     if len(rejected_trades) > 0:
    #         logger.error(f"{len(rejected_trades)} rejected trades.")

    def generate_intervals(self, time_windows):
        """
        time_windows is a list of tuples, each tuple contains the start and end of a time window
        """
        points = set()
        for tw in time_windows:
            points.add(tw[0])
            points.add(tw[1])
        points.add(0)
        
        intervals = []
        # If points is a set, convert it to a sorted list
        points = sorted(list(points))  # Convert set to sorted list
        for i in range(len(points) - 1):
            intervals.append((points[i], points[i+1]))
        intervals.append((points[-1], float('inf')))  
        return intervals
    
    def find_interval_index(self, time, intervals):
        for i, interval in enumerate(intervals, start=1):
            if time >= interval[0] and time < interval[1]:
                return i
        return -1

    def construct_schedule(self, solution, trades, fleets, schedules, scheduled_trades, costs):
        """
        Construct the schedule from the decision variables.
        """
        # maintain a record of the original order of trades
        temp_trades = [id for id in range(len(trades))]
        # create a t,v matrix to record the assignment of trades to vessels
        assignment_matrix = np.zeros((len(trades), len(fleets)))
        for t, v in solution['assignments'].items():
            assignment_matrix[t, v] = 1


        tw_each_vessel = defaultdict(list)


                # current_trade = temp_trades[i]
            
        for v in range(len(fleets)):
            # current_trade = trades[i]
            # is_assigned = False
            current_vessel = fleets[v]
            total_trades_per_vessel = sum(assignment_matrix[:, v])
            while total_trades_per_vessel != 0:
                for i in range(len(trades)):
                    if assignment_matrix[i, v] == 1:
                        is_assigned = True
                    else:
                        continue
                    current_vessel_schedule = schedules.get(current_vessel, current_vessel.schedule)
                    new_schedule = current_vessel_schedule.copy()
                    insertion_points = new_schedule.get_insertion_points()
                    if len(insertion_points) == 1:
                        new_schedule.add_transportation(trades[i])
                        # if new_schedule.verify_schedule():
                        #     schedules[current_vessel] = new_schedule # update the schedule
                        #     scheduled_trades.append(trades[i]) # add the trade to the scheduled trades
                        #     tw_each_vessel[current_vessel].append((solution['pickup_times'][i], solution['dropoff_times'][i]))
                    else:
                        generated_intervals = self.generate_intervals(tw_each_vessel[current_vessel])
                        # raise error if length of generated_intervals is not equal to length of insertion_points, use assert
                        assert len(generated_intervals) == len(insertion_points), "Length of generated intervals is not equal to length of insertion points"
                        pickup_interval_index = self.find_interval_index(solution['pickup_times'][i], generated_intervals)
                        dropoff_interval_index = self.find_interval_index(solution['dropoff_times'][i], generated_intervals)
                        new_schedule.add_transportation(trades[i], pickup_interval_index, dropoff_interval_index)

                    if new_schedule.verify_schedule():
                        schedules[current_vessel] = new_schedule # update the schedule
                        scheduled_trades.append(trades[i]) # add the trade to the scheduled trades
                        tw_each_vessel[current_vessel].append((solution['pickup_times'][i], solution['dropoff_times'][i])) # update the time window for the vessel
                        # calculate the cost of the schedule
                        loading_time = current_vessel.get_loading_time(trades[i].cargo_type, trades[i].amount)
                        loading_cost = current_vessel.get_loading_consumption(loading_time)
                        unloading_costs = current_vessel.get_unloading_consumption(loading_time)
                        travel_time = solution['dropoff_times'][i] - solution['pickup_times'][i]
                        travel_cost = current_vessel.get_laden_consumption(travel_time, current_vessel.speed) # not accurate
                        total_cost = loading_cost + unloading_costs + travel_cost
                        costs[trades[i]] = total_cost * self._profit_factor
                        # remove the trade from the temp_trades
                        total_trades_per_vessel -= 1
                        assignment_matrix[i, v] = 0
                        simple_schedule = schedules[current_vessel].get_simple_schedule()
                        print(f"Vessel {current_vessel} schedule is {simple_schedule}")
            # if schedules.get(current_vessel, current_vessel.schedule).verify_schedule():
                # print(f"Vessel {current_vessel} schedule is valid.")
                # pass
        # return ScheduleProposal(schedules, scheduled_trades, costs)
                    
                    
    def propose_schedules(self, trades):
        schedules = {}
        costs = {}
        scheduled_trades = []
        solver = Solver(self.headquarters)
        solution = solver.solve(trades, self._fleet)
        self.construct_schedule(solution, trades, self._fleet, schedules, scheduled_trades, costs)
        return ScheduleProposal(schedules, scheduled_trades, costs)


