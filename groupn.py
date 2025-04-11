# -*- coding: utf-8 -*-
# @Time    : 09/04/2025 11:27
# @Author  : mmai
# @FileName: groupn
# @Software: PyCharm

from mable.cargo_bidding import TradingCompany

from loguru import logger

class Companyn(TradingCompany):

    def pre_inform(self, trades, time):
        logger.warning("pre_inform")
        _ = self.propose_schedules(trades)

    # def inform(self, trades):
    #     pass
    #
    # def propose_schedules(self, trades):
    #     schedules = {}
    #     scheduled_trades = []