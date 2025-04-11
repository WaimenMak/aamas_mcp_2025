# -*- coding: utf-8 -*-
# @Time    : 07/04/2025 22:23
# @Author  : mmai
# @FileName: test
# @Software: PyCharm

from mable.cargo_bidding import TradingCompany
from mable.examples import environment, fleets
from mable.cargo_bidding import Bid
from mable.transport_operation import ScheduleProposal
# logger
import logging
logger = logging.getLogger(__name__)


# class Companyn(TradingCompany):
#     pass

# class ScheduleProposal:
#     def __init__(self, schedules, scheduled_trades, rejected_trades):
#         self.schedules = schedules
#         self.scheduled_trades = scheduled_trades
#         self.rejected_trades = rejected_trades
#
#     def __str__(self):
#         return f"ScheduleProposal(schedules={self.schedules}, scheduled_trades={self.scheduled_trades}, rejected_trades={self.rejected_trades})"

class Companyn(TradingCompany):
    def pre_inform(self, trades, time):
        pass

    def inform(self, trades, *args, **kwargs):
        proposed_scheduling = self.propose_schedules(trades)
        scheduled_trades = proposed_scheduling.scheduled_trades
        trades_and_costs = [
            (x, -100)
            for x in scheduled_trades
        ]

        bids = [Bid(amount=cost, trade=one_trade) for one_trade, cost in trades_and_costs]
        return bids

    def receive(self, contracts, auction_ledger=None, *args, **kwargs):
        trades = [one_contract.trade for one_contract in contracts]
        scheduling_proposal = self.propose_schedules(trades)
        rejected_trades = self.apply_schedules(scheduling_proposal.schedules)
        if len(rejected_trades) > 0:
            logger.error(f"{len(rejected_trades)} rejected trades.")

    def propose_schedules(self, trades):
        schedules = {}
        scheduled_trades = []
        i = 0
        while i < len(trades):
            current_trade = trades[i]
            is_assigned = False
            j = 0
            while j < len(self._fleet) and not is_assigned:
                current_vessel = self._fleet[j]
                current_vessel_schedule = schedules.get(current_vessel, current_vessel.schedule)
                new_schedule = current_vessel_schedule.copy()
                new_schedule.add_transportation(current_trade)
                if new_schedule.verify_schedule():
                    schedules[current_vessel] = new_schedule
                    scheduled_trades.append(current_trade)
                    is_assigned = True
                j += 1
            i += 1
        return ScheduleProposal(schedules, scheduled_trades, {})

if __name__ == '__main__':
    specifications_builder = environment.get_specification_builder(environment_files_path=".")
    fleet = fleets.example_fleet_1()
    specifications_builder.add_company(Companyn.Data(Companyn, fleet, "My Shipping Corp Ltd."))
    sim = environment.generate_simulation(specifications_builder)
    sim.run()