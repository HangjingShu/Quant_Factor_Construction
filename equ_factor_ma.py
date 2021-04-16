import pandas as pd
import numpy as np

from .base_FG_Equ import Equ


# 股票因子_均线型 20
class EquFactorMa(Equ):
    def __init__(self, OHLC, if_qfq=False):
        super().__init__()
        self.OHLC = OHLC
        self.if_qfq = if_qfq

    def _ACD(self,date_length):
        """
        因子描述：
        6日收集派发指标（6-day Accumulation/Distribution ）。将市场分为两股收集（买入）及派发（估出）的力量
        计算方法：
        1. 若当日收盘价高于昨日收盘价，则收集力量等于当日收盘价与真实低位之差。真实低位是当日低位与昨日收盘价两者中较低者。 buy = close – min(lowest, prev_close)
        2. 若当日收盘价低于昨日收盘价，则派发力量等于当日收盘价与真实高位之差。真实高位是当日高位与昨日收盘价两者中较高者。 sell = close – max(highest, prev_close)
        3. 将收集力量（buy，正数）及派发力量（sell，负数）相加，即可得到市场的净收集力量ACD。ACD = sum(buy,N) + sum(sell,N)
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        # ACD = self.df_OHLC[["close", "low", "high"]].copy()
        # ACD["lag_close"] = ACD["close"].shift(1)
        # ACD["signal"] = ACD["close"] > ACD["lag_close"]
        # ACD["power"] = np.where(ACD["signal"], min(ACD["low"], ACD["lag_close"]), max(ACD["high"], ACD["lag_close"]))
        # ACD["n_signal"] = 2 * ACD["signal"] - 1
        # ACD["n_power"] = ACD["n_signal"] * ACD["power"]
        # ACD_N = ACD["n_power"].rolling(N).sum()
        # return ACD_N
        adj_low = self.df_OHLC['low']
        adj_close = self.df_OHLC['close']
        adj_high = self.df_OHLC['high']
        temp1 = adj_low.mask(adj_close.shift(1) < adj_low, adj_close.shift(1))
        buy = adj_close - temp1
        buy = buy.where(adj_close > adj_close.shift(1), 0)
        temp2 = adj_high.mask(adj_close.shift(1) > adj_high, adj_close.shift(1))
        sell = adj_close - temp2
        sell = sell.where(adj_close < adj_close.shift(1), 0)
        ACD = buy.rolling(window=date_length, center=False).sum() + sell.rolling(window=date_length, center=False).sum()
        return ACD

    def ACD6(self):
        return self._ACD(6)


    def ACD20(self):
        return self._ACD(20)


    def _EMA(self, N, date_length, adj_close):
        """
        因子描述：
        10日指数移动均线（10-day Exponential moving average）。取前N天的收益和当日的价格，当日价格除以（1+当日收益）得到前一日价格，依次计算得到前N日价格，并对前N日价格计算指数移动平均，即为当日的前复权价移动平均
        计算方法：
        1. 若为第一个交易日，EMA即为当日价格
        2. 若非第一个交易日，按如下迭代公式计算：
        其中计算N日EMA时，加权系数 α=2/(N+1)
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        # alpha = 2 / (N + 1)
        # EMA = self.df_OHLC["close"]
        # for i in range(len(EMA)):
        #     if i == 0:
        #         EMA[i] = self.df_OHLC["close"][i]
        #     if i > 0:
        #         EMA[i] = alpha * self.df_OHLC["close"][i] + (1 - alpha) * EMA[i-1]
        # return EMA
        if date_length == 1:
            return adj_close
        else:
            alpha = 2 / (N + 1)
            EMA = alpha * adj_close + (1 - alpha) * self._EMA(N, date_length-1, adj_close.shift(1))
            return EMA


    def EMA10(self):
        adj_close = self.df_OHLC["close"]
        return self._EMA(10, 10, adj_close)

    def EMA12(self):
        adj_close = self.df_OHLC["close"]
        return self._EMA(12, 12, adj_close)

    def EMA120(self):
        adj_close = self.df_OHLC["close"]
        return self._EMA(120, 120, adj_close)

    def EMA20(self):
        adj_close = self.df_OHLC["close"]
        return self._EMA(20, 20, adj_close)

    def EMA26(self):
        adj_close = self.df_OHLC["close"]
        return self._EMA(26, 26, adj_close)

    def EMA5(self):
        adj_close = self.df_OHLC["close"]
        return self._EMA(5, 5, adj_close)

    def EMA60(self):
        adj_close = self.df_OHLC["close"]
        return self._EMA(60, 60, adj_close)

    def _MA(self,date_length):
        """
        因子描述：
        移动均线（Moving average）。取最近N天的动态前复权价格的均值
        """
        adj_close = self.df_OHLC["close"]
        # MA_D = MA["close"].rolling(N).mean()
        MA = adj_close.rolling(window=date_length, center=False).mean()
        return MA

    def MA120(self):
        return self._MA(120)

    def MA20(self):
        return self._MA(20)

    def MA10(self):
        return self._MA(10)

    def MA5(self):
        return self._MA(5)

    def MA60(self):
        return self._MA(60)

    def APBMA(self):
        """
        因子描述：
        绝对偏差移动平均（Absolute Price Bias Moving Average）。考察一段时期内价格偏离均线的移动平均。
        计算方法：
        APBMA = MA(abs(close – MA(close, N)), N)
        其中N = 5。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        close = self.df_OHLC["close"]
        MA5 = self._MA(5)
        diff = abs(close - MA5)
        APBMA = diff.rolling(5).mean()
        return APBMA

    def BBI(self):
        """
        因子描述：
        多空指数（Bull and Bear Index）。是一种将不同日数移动平均线加权平均之后的综合指标，属于均线型指标。
        计算方法：
        BBI = (MA3 + MA6 + MA12 + MA24) / 4
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        MA3 = self._MA(3)
        MA6 = self._MA(6)
        MA12 = self._MA(12)
        MA24 = self._MA(24)
        BBI = (MA3 + MA6 + MA12 + MA24) / 4
        return BBI

    def BBIC(self):
        """
        因子描述：
        因子BBI除以收盘价得到 (BBI/Close price) 。
        计算方法：
        BBIC = BBI / close
        """
        close = self.df_OHLC["close"]
        BBI = self._BBI()
        BBIC = BBI / close
        return BBIC

    def _TEMA(self,N):
        """
        因子描述：
        10日三重指数移动平均线（10-day Triple Exponential Moving Average）。取时间N内的收盘价分别计算其一至三重指数加权平均。
        计算方法：
        TEMA = 3 * 一重指数加权平均 – 3 * 二重指数加权平均 + 三重指数加权平均；
        TEMA = 3 * EMA(close, N) – 3 * EMA(EMA(close, N), N) + EMA(EMA(EMA(close, N), N), N)；
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        # EMA = self._MA(N)
        # alpha = 2 / (N + 1)
        # EMA_D = EMA.copy()
        # EMA_T = EMA_D.copy()
        # for i in range(len(EMA)):
        #     if i == 0:
        #         EMA_D[i] = EMA[i]
        #         EMA_T[i] = EMA_D[i]
        #     if i > 0:
        #         EMA_D[i] = alpha * EMA[i] + (1 - alpha) * EMA_D[i-1]
        #         EMA_T[i] = alpha * EMA_D[i] + (1-alpha) * EMA_T[i-1]
        # TEMA = 3 * EMA - 3 * EMA_D + EMA_T
        # return TEMA
        adj_close = self.df_OHLC["close"]
        TEMA = 3 * self._EMA(N, N, adj_close) - 3 * self._EMA(N, N, self._EMA(N, N, adj_close)) + \
               self._EMA(N, N, self._EMA(N, N, self._EMA(N, N, adj_close)))
        return TEMA


    def TEMA10(self):
        return self._TEMA(10)

    def TEMA5(self):
        return self._TEMA(5)

    def MA10Close(self):
        """
        因子描述：
        均线价格比 (10-day moving average to close price ratio)。由于股票的成交价格有响起均线回归的趋势，计算均线价格比可以预测股票在未来周期的运动趋势。
        计算方法：
        MA10Close = MA(收盘价, N) / 收盘价
        N取10
        """
        MA = self._MA(10)
        close = self.df_OHLC["close"]
        MA10Close = MA / close
        return MA10Close

