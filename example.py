# -*- coding: utf-8 -*-
# @Time    : 09/04/2025 11:26
# @Author  : mmai
# @FileName: main
# @Software: PyCharm


from mable.examples import environment, fleets, companies
# from Agents import Solver
import greedy
# import kbest
import kbest_bid
import group7

def build_specification():
    number_of_month = 12
    trades_per_auction = 50
    num = 2 # number of vessels per fleet
    specifications_builder = environment.get_specification_builder(
        trades_per_occurrence=trades_per_auction,
        num_auctions=number_of_month)

    # solver
    # my_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
    # specifications_builder.add_company(groupn.OurCompanyn.Data(groupn.OurCompanyn, my_fleet, groupn.OurCompanyn.__name__, profit_factor=1.5))

    # greedy
    my_fleet = fleets.mixed_fleet(num_suezmax=num, num_aframax=num, num_vlcc=num)
    # my_fleet = fleets.mixed_fleet(num_suezmax=num, num_vlcc=num)
    specifications_builder.add_company(greedy.GreedyComanyn.Data(greedy.GreedyComanyn, my_fleet, greedy.GreedyComanyn.__name__, profit_factor=1.4))


    # kbest
    # my_fleet = fleets.mixed_fleet(num_suezmax=num)
    # specifications_builder.add_company(kbest.KBestComanyn.Data(kbest.KBestComanyn, my_fleet, kbest.KBestComanyn.__name__, profit_factor=1.4))

    # # kbest bid
    my_fleet = fleets.mixed_fleet(num_suezmax=num, num_aframax=num, num_vlcc=num)
    specifications_builder.add_company(kbest_bid.KBestBidComanyn.Data(
        kbest_bid.KBestBidComanyn,
        my_fleet,
        kbest_bid.KBestBidComanyn.__name__,
        profit_factor=1.4,          # < absolute cost
        profit_factor_2=1.2,        # > absolute cost
        avg_w=0.7,                  # avg_w for average cost, 1 - avg_w for absolute cost
        cal_efficiency=True,       # calculate efficiency
        schedule_with_greedy=False, # schedule with greedy
        efficiency_selection_percentage=0.8,    # percentage of trades to consider for efficiency selection
        trade_frequency_threshold=0.5,          # threshold for trade frequency
        k_best=110,                             # number of best schedules to consider
        runtime_limit=57,                       # runtime limit
        pruning_factor=1.2))                      # pruning factor

    # group7
    my_fleet = fleets.mixed_fleet(num_suezmax=num, num_aframax=num, num_vlcc=num)
    specifications_builder.add_company(group7.Company7.Data(group7.Company7, my_fleet, group7.Company7.__name__))

    # arch enemy
    arch_enemy_fleet = fleets.mixed_fleet(num_suezmax=num, num_aframax=num, num_vlcc=num)
    specifications_builder.add_company(
        companies.MyArchEnemy.Data(
            companies.MyArchEnemy, arch_enemy_fleet, "Arch Enemy Ltd.",
            profit_factor=1.5))

    # arch enemy 2
    # arch_enemy_fleet = fleets.mixed_fleet(num_suezmax=num, num_aframax=num, num_vlcc=num)
    # specifications_builder.add_company(
    #     companies.MyArchEnemy.Data(
    #         companies.MyArchEnemy, arch_enemy_fleet, "Arch Enemy Ltd.2",
    #         profit_factor=2.0))

    # scheduler fleet
    the_scheduler_fleet = fleets.mixed_fleet(num_suezmax=num, num_aframax=num, num_vlcc=num)
    specifications_builder.add_company(
        companies.TheScheduler.Data(
            companies.TheScheduler, the_scheduler_fleet, "The Scheduler LP",
            profit_factor=1.4))

    sim = environment.generate_simulation(
        specifications_builder,
        show_detailed_auction_outcome=False,
        global_agent_timeout=60)
    sim.run()


if __name__ == '__main__':
    build_specification()