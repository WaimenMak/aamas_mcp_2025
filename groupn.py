# -*- coding: utf-8 -*-
# @Time    : 09/04/2025 11:27
# @Author  : mmai
# @FileName: groupn
# @Software: PyCharm

from mable.cargo_bidding import TradingCompany, SimpleCompany
from ortools.sat.python import cp_model

from loguru import logger
from mable.cargo_bidding import Bid
from copy import deepcopy
from math import ceil

class Solver:
    def __init__(self, headquarters):
        self.headquarters = headquarters

    def solve(self, trades, fleets):
        """
        Solve the problem of scheduling the trades. Input is a list of trades and output decision variables.
        time_step is the time step of current time
        """
        # process the trades and assign a unique id to each trade
        start_time = trades[0].time
        trades_with_id = []
        for i, trade in enumerate(trades):
            trade_with_id = deepcopy(trade)
            setattr(trade_with_id, "id", i)
            # add travel time to the trade
            travel_distance = self.headquarters.get_network_distance(trade.origin_port, trade.destination_port)
            # travel_time = fleets[0].get_travel_time(travel_distance)
            setattr(trade_with_id, "travel_distance", travel_distance)
            trades_with_id.append(trade_with_id)

        model = cp_model.CpModel()
        # define decision variables
        assign = {}
        pickup_time = {}
        dropoff_time = {}
        for t, trade in enumerate(trades):
            for v, vessel in enumerate(fleets):
                assign[t, v] = model.NewBoolVar(f"assign_t{t}_v{v}")
            earliest_pickup = trade.time_window[0]
            latest_pickup = trade.time_window[1]
            earliest_dropoff = trade.time_window[2]
            latest_dropoff = trade.time_window[3]

            pickup_time[t] = model.NewIntVar(earliest_pickup, latest_pickup, f"pickup_time_t{t}")
            dropoff_time[t] = model.NewIntVar(earliest_dropoff, latest_dropoff, f"dropoff_time_t{t}")

        # define constraints
        # Constraint: each trade is either served by one vessel or unserved
        for t in range(len(trades)):
            model.Add(sum(assign[t, v] for v in range(len(fleets))) <= 1)

        # Constraint: the pickup time must be before the dropoff time, hard constraint
        for t, trade in enumerate(trades):
            for v, vessel in enumerate(fleets):
                loading_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
                # unloading_time = loading_time
                # travel_distance = self.headquarters.get_network_distance(trade.origin_port, trade.destination_port)
                travel_distance = trades_with_id[t].travel_distance
                travel_time = vessel.get_travel_time(travel_distance)
                journey_duration = ceil(travel_time + loading_time)
                setattr(trades_with_id[t], "duration", journey_duration)
                # Create an interval for the journey
                # journey_interval = model.NewIntervalVar(
                #     pickup_time[t],               # start
                #     journey_duration,             # duration
                #     pickup_time[t] + journey_duration,  # end
                #     f'journey_{t}_{v}'            # name
                # )
                
                # Add constraint that dropoff time must be after journey end
                model.Add(pickup_time[t] + journey_duration <= dropoff_time[t]).OnlyEnforceIf(assign[t, v])
                model.Add(pickup_time[t] >= trade.time_window[0]).OnlyEnforceIf(assign[t, v]) # pickup time must be after the earliest pickup time
                model.Add(pickup_time[t] <= trade.time_window[1]).OnlyEnforceIf(assign[t, v]) # pickup time must be before the latest pickup time
                model.Add(dropoff_time[t] >= trade.time_window[2]).OnlyEnforceIf(assign[t, v]) # dropoff time must be after the earliest dropoff time
                model.Add(dropoff_time[t] <= trade.time_window[3]).OnlyEnforceIf(assign[t, v]) # dropoff time must be before the latest dropoff time

        # Constraint: capacity constraint
        # for v, vessel in enumerate(fleets):
        #     model.Add(sum(assign[t, v] * trade.amount for t, trade in enumerate(trades)) <= vessel.capacity)
        for v, vessel in enumerate(fleets):
            intervals = []
            demands = []
            for t, trade in enumerate(trades):
                interval = model.NewOptionalIntervalVar(
                    pickup_time[t], 
                    trades_with_id[t].duration,  # duration: loading time + travel time
                    dropoff_time[t],
                    assign[t, v],
                    f'interval_{t}_{v}'
                )
                intervals.append(interval)
                demands.append(ceil(trade.amount))
            capacity_list = vessel.capacities_and_loading_rates
            model.AddCumulative(intervals, demands, ceil(capacity_list[0].capacity))

        # Modelling the idle time and ballast time
        earliest_pickup = min(trade.time_window[0] for trade in trades)
        latest_dropoff = max(trade.time_window[3] for trade in trades)
        max_time = latest_dropoff - earliest_pickup

        idle_consumption_expr = []
        ballast_consumption_expr = []
        
        # Add calculations for the initial positioning of each vessel
        for v, vessel in enumerate(fleets):
            vessel_location = vessel.location  # Current location of the vessel
            # For each potential first trade for this vessel
            for t in range(len(trades)):
                # Create a variable that indicates if trade t is the first for vessel v
                is_first_trade = model.NewBoolVar(f"first_trade_{t}_v{v}")
                # This is the first trade for vessel v if:
                # 1. It's assigned to vessel v
                # 2. No other trade with an earlier pickup time is assigned to vessel v
                model.Add(is_first_trade <= assign[t, v])  # Can only be first if assigned
                # Check if this is the earliest assigned trade
                for other_t in range(len(trades)):
                    if other_t == t:
                        continue
                    # If other_t has earlier pickup time and is assigned to v, then t is not first
                    earlier_pickup = model.NewBoolVar(f"earlier_pickup_{other_t}_than_{t}_v{v}")
                    model.Add(pickup_time[other_t] < pickup_time[t]).OnlyEnforceIf(earlier_pickup)
                    model.Add(pickup_time[other_t] >= pickup_time[t]).OnlyEnforceIf(earlier_pickup.Not())
                    # If other_t is earlier and assigned, t is not first
                    not_first_due_to_other = model.NewBoolVar(f"not_first_due_to_{other_t}_{t}_v{v}")
                    model.AddBoolAnd([earlier_pickup, assign[other_t, v]]).OnlyEnforceIf(not_first_due_to_other)
                    model.AddBoolOr([earlier_pickup.Not(), assign[other_t, v].Not()]).OnlyEnforceIf(not_first_due_to_other.Not())
                    model.Add(is_first_trade <= 1 - not_first_due_to_other)
                
                # Calculate ballast distance from vessel's initial position to first trade pickup
                initial_travel_distance = self.headquarters.get_network_distance(vessel_location, trades[t].origin_port)
                initial_travel_time = vessel.get_travel_time(initial_travel_distance)
                
                # Add ballast consumption for this initial positioning
                initial_ballast_consumption = vessel.get_ballast_consumption(initial_travel_time, vessel.speed)
                
                # Create a temporary variable to hold this consumption if this is the first trade
                temp_ballast_var = model.NewIntVar(0, ceil(initial_ballast_consumption * 10), f"first_ballast_{t}_v{v}")
                model.Add(temp_ballast_var == ceil(initial_ballast_consumption * 10)).OnlyEnforceIf(is_first_trade)
                model.Add(temp_ballast_var == 0).OnlyEnforceIf(is_first_trade.Not())
                
                ballast_consumption_expr.append(temp_ballast_var)
        
        # # Original code for subsequent trips
        for v, vessel in enumerate(fleets):
            for t1 in range(len(trades)):
                for t2 in range(len(trades)):
                    if t1 == t2:
                        continue
                    # Bool: Are t1 and t2 both assigned to v
                    both_assigned = model.NewBoolVar(f"pair_{t1}_{t2}_v{v}")
                    model.AddBoolAnd([assign[t1, v], assign[t2, v]]).OnlyEnforceIf(both_assigned)
                    model.AddBoolOr([assign[t1, v].Not(), assign[t2, v].Not()]).OnlyEnforceIf(both_assigned.Not())
                    # add ordering condition:
                    t2_after_t1 = model.NewBoolVar(f"t2_after_t1_{t1}_{t2}_v{v}")
                    model.Add(pickup_time[t2] >= dropoff_time[t1] + 1).OnlyEnforceIf(t2_after_t1)
                    model.Add(pickup_time[t2] < dropoff_time[t1] + 1).OnlyEnforceIf(t2_after_t1.Not())
                    #compute gap time check if it exceeds unloading time + travel time
                    gap_time = model.NewIntVar(0, max_time, f"gap_{t1}_{t2}_v{v}")
                    model.Add(gap_time == pickup_time[t2] - dropoff_time[t1])
                    # gap between t1 and t2 must be at least unloading time + travel time
                    travel_distance = self.headquarters.get_network_distance(trades[t1].destination_port, trades[t2].origin_port)
                    travel_time = ceil(vessel.get_travel_time(travel_distance))
                    min_gap_required = ceil(vessel.get_loading_time(trades[t1].cargo_type, trades[t1].amount) + travel_time)
                    model.Add(gap_time >= min_gap_required).OnlyEnforceIf(both_assigned)
                    gap_ok = model.NewBoolVar(f"gap_ok_{t1}_{t2}_v{v}")
                    model.Add(gap_time >= min_gap_required).OnlyEnforceIf(gap_ok)
                    model.Add(gap_time < min_gap_required).OnlyEnforceIf(gap_ok.Not())

                    idle_time = model.NewIntVar(0, max_time, f"idle_{t1}_{t2}_v{v}")
                    ballast_time = model.NewIntVar(0, max_time, f"ballast_{t1}_{t2}_v{v}")

                    model.Add(idle_time == gap_time - min_gap_required).OnlyEnforceIf([
                        both_assigned, t2_after_t1, gap_ok
                    ])
                    model.Add(idle_time == 0).OnlyEnforceIf([
                        both_assigned.Not()
                    ]).OnlyEnforceIf([
                        t2_after_t1.Not()
                    ]).OnlyEnforceIf([
                        gap_ok.Not()
                    ])
                    # idle_consumption_expr.append(vessel.get_idle_consumption(idle_time))
                    idle_consumption_rate = ceil(vessel._propelling_engine._idle_consumption)
                    idle_consumption_expr.append(idle_time * idle_consumption_rate) # linear approximation
                    model.Add(ballast_time == travel_time).OnlyEnforceIf([
                        both_assigned, t2_after_t1, gap_ok
                    ])
                    # ballast_consumption_expr.append(vessel.get_ballast_consumption(ballast_time, vessel.speed))
                    base = ceil(vessel.Data.ballast_consumption_rate.base)
                    factor = ceil(vessel.Data.ballast_consumption_rate.factor)
                    pow_speed = ceil(pow(vessel.speed, vessel.Data.ballast_consumption_rate.speed_factor))
                    ballast_consumption_expr.append(base * pow_speed * ballast_time * factor) # linear approximation

        # max_travel_distance = max(trades_with_id[t].travel_distance for t in range(len(trades)))
        # min_fleet_speed = min(vessel.speed for vessel in fleets)
        # max_travel_time = max_travel_distance / min_fleet_speed
        # max_dropoff_time = max(trade.time_window[3] for trade in trades)
        # max_idle_consumption = len(fleets) * fleets[0].get_idle_consumption(max_dropoff_time - start_time)

        # max_ballast_legs = len(fleets) * len(trades)
        # max_ballast_consumption = max_ballast_legs * fleets[0].get_ballast_consumption(max_travel_time, fleets[0].speed)

        # big_M = ceil(max_idle_consumption + max_ballast_consumption)
        # total_idle_cost = model.NewIntVar(0, big_M, "total_idle_cost")
        # total_ballast_cost = model.NewIntVar(0, big_M, "total_ballast_cost")

        # model.Add(total_idle_cost == sum(idle_consumption_expr))
        # model.Add(total_ballast_cost == sum(ballast_consumption_expr))

        # Objective: minimize the total cost
        fuel_expr = []
        penalty_expr = []
        for t, trade in enumerate(trades):
            for v, vessel in enumerate(fleets):
                travel_distance = self.headquarters.get_network_distance(trade.origin_port, trade.destination_port)
                loading_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
                loading_cost = vessel.get_loading_consumption(loading_time)
                unloading_costs = vessel.get_unloading_consumption(loading_time)
                travel_time = vessel.get_travel_time(travel_distance)
                travel_cost = vessel.get_laden_consumption(travel_time, vessel.speed)
                total_cost = loading_cost + unloading_costs + travel_cost
                fuel_expr.append(assign[t, v] * total_cost)
            penalty_expr.append((1 - sum(assign[t, v] for v in range(len(fleets)))) * trade.amount) # trade.amount is the penalty for unserved trades

        # idle cost
        total_idle_cost = sum(idle_consumption_expr)
        total_ballast_cost = sum(ballast_consumption_expr)
        # model.Minimize(sum(fuel_expr) + total_idle_cost + total_ballast_cost + sum(penalty_expr))
        model.Minimize(sum(fuel_expr) + sum(penalty_expr) + total_ballast_cost + total_idle_cost)
        # solve the problem
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print("Solution:")
            for t, trade in enumerate(trades):
                served = False
                for v, vessel in enumerate(fleets):
                    if solver.Value(assign[t, v]):
                        served = True
                        print(f"Trade {t} is served by vessel {v}")
                        print(f"  Pickup:  {solver.Value(pickup_time[t])} at port {trade.origin_port}")
                        print(f"  Dropoff: {solver.Value(dropoff_time[t])} at port {trade.destination_port}")
                if not served:
                    print(f"Trade {t} is unserved (penalty {trade.amount})")
            print("Total cost:", solver.ObjectiveValue())
            return solver.Value
        else:
            print("No solution found.")
            return None


    def construct_schedule(self, decision_variables):
        """
        Construct the schedule from the decision variables.
        """
        pass


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
    
    def propose_schedules(self, trades):
        schedules = {}
        scheduled_trades = []
        solver = Solver(self.headquarters)
        assign, pickup_time, dropoff_time = solver.solve(trades, self._fleet)
        for t, trade in enumerate(trades):
            schedules[trade.id] = {
                "pickup": (pickup_time[t], dropoff_time[t]),
                "dropoff": (pickup_time[t], dropoff_time[t])
            }


