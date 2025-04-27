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

from mable.cargo_bidding import TradingCompany
from mable.examples.companies import ScheduleProposal
import attrs
from marshmallow import fields
import time
import random
from utils import simulate_schedule_cost_allocated_shared_arrival, simulate_schedule_cost
random.seed(1)
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
            if time_end - time_start > 55: # 50 seconds timeout
                break
        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")

        # get the minimum cost schedule
        # if len(k_best_schedule_costs) != 0:
        #     min_cost_schedule_index = k_best_schedule_costs.index(min(k_best_schedule_costs))
        #     schedules = k_best_schedules[min_cost_schedule_index]
        # -- bid based on average cost of k best schedules
        if len(k_best_schedules) != 0:
            trade_frequencies, trade_avg_costs = self.calculate_trade_frequency_and_avg_cost(
                k_best_schedules,
                len(k_best_schedules),
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
                bid_price = 0.8 * avg_cost + 0.2 * absolute_cost
                if bid_price < absolute_cost:
                    costs[trade] = bid_price * self._profit_factor
                else:
                    costs[trade] = bid_price * 1.2
                scheduled_trades.append(trade)


        # return ScheduleProposal(schedules, scheduled_trades, costs)
        return ScheduleProposal({}, scheduled_trades, costs)

    def receive(self, contracts, auction_ledger=None, *args, **kwargs):
        trades = [one_contract.trade for one_contract in contracts]
        payment_per_trade = {}
        for one_contract in contracts:
            payment_per_trade[one_contract.trade] = one_contract.payment
        scheduling_proposal = self.schedule_trades(trades, payment_per_trade)
        # --- Use GreedyComanyn's propose_schedules logic ---
        # 1. Create a temporary instance of GreedyComanyn using this company's fleet/hq/etc.
        #    (Assumes __init__ signatures are compatible or GreedyComanyn doesn't need specific state)
        # temp_greedy_company = GreedyComanyn(self._fleet, self.name, self._profit_factor)
        # 2. Set up the headquarters for the temporary instance if needed (standard pattern)
        # temp_greedy_company._headquarters = self._headquarters
        # 3. Call the propose_schedules method on the temporary instance
        # scheduling_proposal = temp_greedy_company.propose_schedules(trades, payment_per_trade)
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
            end_time = time.time()
            if end_time - start_time > 55:
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


