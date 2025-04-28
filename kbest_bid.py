# -*- coding: utf-8 -*-
# @Time    : 24/04/2025 23:19
# @Author  : mmai
# @FileName: kbest_bid
# @Software: PyCharm

from mable.cargo_bidding import TradingCompany
from mable.examples.companies import ScheduleProposal
import attrs
from marshmallow import fields
import time
import random
from utils import simulate_schedule_cost_allocated_shared_arrival, simulate_schedule_cost, cal_efficiency
# random.seed(1)
from greedy import GreedyComanyn # Added alias if needed

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
    def __init__(self, fleet, name, profit_factor=1.65, profit_factor_2=1.2, 
                 avg_w=0.7, cal_efficiency=False, schedule_with_greedy=False,
                 efficiency_selection_percentage=0.8, trade_frequency_threshold=0.5, 
                 k_best=110, runtime_limit=55, pruning_factor=1):
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
        profit_factor: float = 1.65
        profit_factor_2: float = 1.2
        avg_w: float = 0.7
        cal_efficiency: bool = False
        schedule_with_greedy: bool = False
        efficiency_selection_percentage: float = 0.8
        trade_frequency_threshold: float = 0.5
        k_best: int = 110
        runtime_limit: int = 55
        pruning_factor: float = 1
        class Schema(TradingCompany.Data.Schema):
            profit_factor = fields.Float(default=1.65)
            profit_factor_2 = fields.Float(default=1.2)
            avg_w = fields.Float(default=0.7)
            cal_efficiency = fields.Boolean(default=False)
            schedule_with_greedy = fields.Boolean(default=False)
            efficiency_selection_percentage = fields.Float(default=0.8)
            trade_frequency_threshold = fields.Float(default=0.5)
            k_best = fields.Integer(default=110)
            runtime_limit = fields.Integer(default=55)
            pruning_factor = fields.Float(default=1)
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
                # if time.time() - start_execution_time > 50:
                #     break
                
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
        # for v, vessel in enumerate(self._fleet):
        #     if len(vessel.schedule.get_simple_schedule()) ==3:
        #         pass
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
                # optional: calculate the cost of the schedule
                # schedule_cost = get_costs_for_schedule(schedule, self._fleet, self._headquarters, start_time)
                # k_best_schedule_costs.append(schedule_cost)

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
            # get the minimum cost schedule
            # if len(k_best_schedule_costs) != 0:
            #     min_cost_schedule_index = k_best_schedule_costs.index(min(k_best_schedule_costs))
            #     schedules = k_best_schedules[min_cost_schedule_index]
            # -- bid based on average cost of k best schedules
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
                    costs[trade] = bid_price * self._profit_factor
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

        if not self.schedule_with_greedy:
            scheduling_proposal = self.schedule_trades(trades, payment_per_trade)
        else:
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
                # cost, _, _, _, _ = simulate_schedule_cost_allocated_shared_arrival(
                #     vessel,
                #     schedule,
                #     start_time,
                #     self._headquarters,
                #     payment_per_trade)    # there is a bug in this function, but for bidding it is ok
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


