from .base_FG_Equ import Equ
import pandas as pd
import numpy as np

"""
get series from df_bic
calculate if necessary
drop duplicated records
expand the index list of df_bic by df_sb
div df_sb if necessary
"""


# 股票因子_成交量型 34
class EquFactorVolume(Equ):
    def __init__(self, OHLC, sb, if_qfq=False):
        super().__init__()
        self.OHLC = OHLC
        self.sb = sb
        self.if_qfq = if_qfq

    def _EMA(self, N, date_length, adj_close):
        if date_length == 1:
            return adj_close
        else:
            alpha = 2 / (N + 1)
            EMA = alpha * adj_close + (1 - alpha) * self._EMA(N, date_length - 1, adj_close.shift(1))
            return EMA

    def KlingerOscillator(self):
        """
        因子描述：
        成交量摆动指标 (Klinger Oscillator)。该指标在决定长期资金流量趋势的同时保持了对于短期资金流量的敏感性，因而
        可以用于预测短期价格拐点。
        计算方法：
        1.典型价格TYP = (close + highest + lowest) / 3。
        2.若今日典型价格高于昨日典型价格则成交量记为正，反之记为负。然后计算34日和55日内经上述处理过的成交量的指
        数平均数，将两平均数相减得到差值。
        3.计算该差值的6日指数平均值，最后入库数值除以1e6。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        close = self.df_OHLC["close"]
        highest = self.df_OHLC["high"]
        lowest = self.df_OHLC["low"]
        volume = self.df_OHLC["volume"]
        # TYP = (close + highest + lowest) / 3
        # lag_TYP = TYP.shift(1)
        # signal = 2 * (TYP > lag_TYP) - 1
        # signal_volume = signal * volume
        # ema34 = self._EMA(6, 6, signal_volume)
        TYP = (close + highest + lowest) / 3
        temp = TYP.diff(1)
        adj_volume = volume.where(temp > 0, -volume)
        KO = self._EMA(6, 6, (self._EMA(34, 34, adj_volume) - self._EMA(55, 55, adj_volume))) / 1e6
        return KO

    def MoneyFlow20(self):
        """
        因子描述：
        资金流量 (20-day money flow)。用收盘价，最高价及最低价的均值乘以当日成交量即可得到该交易日的资金流量。
        计算方法：
        此处取N = 20。
        """
        close = self.df_OHLC["close"]
        highest = self.df_OHLC["high"]
        lowest = self.df_OHLC["low"]
        volume = self.df_OHLC["volume"]
        moneyflow = (close + highest + lowest) * volume / 3
        MoneyFlow20 = moneyflow.rolling(20, 1).sum()
        return MoneyFlow20

    def OBV(self):
        """
        因子描述：
        能量潮指标（On Balance Volume，OBV）。以股市的成交量变化来衡量股市的推动力，从而研判股价的走势。
        计算方法：
        计算累积成交量。如果本日收盘价或指数高于前一日收盘价或指数，本日值则为正；如果本日的收盘价或指数低于前一日
        的收盘价，本日值则为负值；如果本日值与前一日的收盘价或指数持平，本日值则不予计算，然后计算累积成交量。这里
        的成交量是指成交股票的手数。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        close = self.df_OHLC["close"]
        volume = self.df_OHLC["volume"]
        lag_close = close.shift(1)
        signal = close.copy()
        for i in range(len(close)):
            if close[i] > lag_close[i]:
                signal[i] = 1
            if close[i] == lag_close[i]:
                signal[i] = np.nan
            if close[i] < lag_close[i]:
                signal[i] = -1
        signal_volume = volume * signal
        OBV = signal_volume.cumsum()
        return OBV

    def OBV20(self):
        """
        因子描述：
        20日能量潮指标（20-day On Balance Volume，OBV）。以股市的成交量变化来衡量股市的推动力，从而研判股价的
        走势。
        计算方法
        计算20日累积成交量的均值。如果本日收盘价或指数高于前一日收盘价或指数，本日值则为正；如果本日的收盘价或指
        数低于前一日的收盘价，本日值则为负值；如果本日值与前一日的收盘价或指数持平，本日值则不予计算，然后计算20
        日累积成交量。这里的成交量是指成交股票的手数。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        close = self.df_OHLC["close"]
        volume = self.df_OHLC["volume"]
        lag_close = close.shift(1)
        signal = close.copy()
        for i in range(len(close)):
            if close[i] > lag_close[i]:
                signal[i] = 1
            if close[i] == lag_close[i]:
                signal[i] = np.nan
            if close[i] < lag_close[i]:
                signal[i] = -1
        signal_volume = volume * signal
        OBV20 = signal_volume.rolling(20, 1).sum()
        return OBV20

    def OBV6(self):
        """
        因子描述：
        6日能量潮指标（6-day On Balance Volume，OBV）。以股市的成交量变化来衡量股市的推动力，从而研判股价的走
        势。
        计算方法：
        计算6日累积成交量的均值。如果本日收盘价或指数高于前一日收盘价或指数，本日值则为正；如果本日的收盘价或指数
        低于前一日的收盘价，本日值则为负值；如果本日值与前一日的收盘价或指数持平，本日值则不予计算，然后计算6日累
        积成交量。这里的成交量是指成交股票的手数。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        close = self.df_OHLC["close"]
        volume = self.df_OHLC["volume"]
        lag_close = close.shift(1)
        signal = close.copy()
        for i in range(len(close)):
            if close[i] > lag_close[i]:
                signal[i] = 1
            if close[i] == lag_close[i]:
                signal[i] = np.nan
            if close[i] < lag_close[i]:
                signal[i] = -1
        signal_volume = volume * signal
        OBV6 = signal_volume.rolling(20, 1).sum()
        return OBV6

    def DAVOL10(self):
        """
        因子描述：
        相对10日相对120日平均换手率（Difference between 10-day average turnover rate and 120 -day average
        turnover rate）。
        计算方法：
        最近一段时间的日平均换手率相对于120个交易日的变化。计算了10个交易日的结果。
        """
        turnover_rate = self.df_sb["turnover_rate"]
        TR10 = turnover_rate.rolling(10, 1).mean()
        TR120 = turnover_rate.rolling(120, 1).mean()
        DAVOL10 = TR10 - TR120
        return DAVOL10

    def DAVOL20(self):
        """
        因子描述：
        相对20日相对120日平均换手率（Difference between 20-day average turnover rate and 120 -day average
        turnover rate）。
        计算方法：
        最近一段时间的日平均换手率相对于120个交易日的变化。计算了20个交易日的结果。
        """
        turnover_rate = self.df_sb["turnover_rate"]
        TR20 = turnover_rate.rolling(window=20, min_periods=1).mean()
        TR120 = turnover_rate.rolling(window=120, min_periods=1).mean()
        DAVOL20 = TR20 - TR120
        return DAVOL20

    def DAVOL5(self):
        """
        因子描述：
        相对5日相对120日平均换手率（Difference between 5-day average turnover rate and 120 -day average turnover
        rate）。
        计算方法：
        最近一段时间的日平均换手率相对于120个交易日的变化。计算了5个交易日的结果。
        """
        turnover_rate = self.df_sb["turnover_rate"]
        TR5 = turnover_rate.rolling(5, 1).mean()
        TR120 = turnover_rate.rolling(120, 1).mean()
        DAVOL5 = TR5 - TR120
        return DAVOL5

    def _DIFF(self):
        adj_close = self.df_OHLC['close']
        DIFF = self._EMA(12, 12, adj_close) - self._EMA(26, 26, adj_close)
        return DIFF

    def _DEA(self):
        DEA = self._EMA(9, 9, self._DIFF())
        return DEA

    def VDEA(self):
        """
        因子描述：
        计算VMACD因子的中间变量（Volume Difference Exponential Average）。
        计算方法：
        VDIFF = EMA(volume, 12) - EMA(volume, 26)
        DEA为VDIFF的M日指数移动平均：DEA = EMA(VDIFF, M)，通常M = 9。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        VDIFF = self.VDIFF()
        VDEA = self._EMA(9, 9, VDIFF)
        return VDEA

    def VDIFF(self):
        """
        因子描述：
        计算VMACD因子的中间变量 (Volume difference)。
        计算方法：
        VDIFF = EMA(volume, 12) - EMA(volume, 26)
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        adj_volume = self.df_OHLC['volume']
        VDIFF = self._EMA(12, 12, adj_volume) - self._EMA(26, 26, adj_volume)
        return VDIFF

    def _VEMA(self, date_length):
        adj_volume = self.df_OHLC['volume']
        VEMA = self._EMA(date_length, date_length, adj_volume)
        return VEMA

    def VEMA10(self):
        """
        因子描述：
        成交量的指数移动平均 (10-day Exponential moving average of volume)。
        计算方法：
        VEMA = EMA(volume, N)
        N取10。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        return self._VEMA(10)

    def VEMA12(self):
        """
        因子描述：
        成交量的指数移动平均(12-day Exponential moving average of volume)。
        计算方法：
        VEMA = EMA(volume, N)
        N取12。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        return self._VEMA(10)

    def VEMA26(self):
        """
        因子描述：
        成交量的指数移动平均 (26-day Exponential moving average of volume) 。
        计算方法：
        VEMA = EMA(volume, N)
        N取26。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        return self._VEMA(26)

    def VEMA5(self):
        """
        因子描述：
        成交量的指数移动平均 (5-day Exponential moving average of volume) 。
        计算方法：
        VEMA = EMA(volume, N)
        N取5。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        return self._VEMA(5)

    def VMACD(self):
        """
        因子描述：
        量指数平滑异同移动平均线（Volume Moving Average Convergence and Divergence）。是从双移动平均线发展而
        来的，由快的移动平均线减去慢的移动平均线, VMACD的意义和MACD基本相同, 但VMACD取用的数据源是成交
        量，MACD取用的数据源是成交价格。
        计算方法：
        1. VDIFF = EMA(volume, 12) - EMA(volume, 26)。
        2. VDEA为（Difference Exponential Average）VDIFF的M日指数移动平均：DEA = EMA(VDIFF, M)，通常M = 9。
        3. VMACD为VDIFF和VDEA之差 * 2。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        VDIFF = self.VDIFF()
        VDEA = self.VDEA()
        VMACD = (VDIFF - VDEA) * 2
        return VMACD

    def VOL10(self):
        """
        因子描述：
        10日平均换手率（10-day Turnover Rate）。
        计算方法：
        最近10个交易日的平均换手率。
        注：取个股最近N个非停牌日进行计算
        """
        Turnover = self.df_sb["turnover_rate"]
        turnover10 = Turnover.rolling(10, 1).mean()
        return turnover10

    def VOL120(self):
        """
        因子描述：
        120日平均换手率（120-day Turnover Rate）。
        计算方法：
        最近120个交易日的平均换手率。
        """
        Turnover = self.df_sb["turnover_rate"]
        turnover120 = Turnover.rolling(120, 1).mean()
        return turnover120

    def VOL20(self):
        """
        因子描述：
        20日平均换手率（20-day Turnover Rate）。
        计算方法：
        最近20个交易日的平均换手率。
        注：取个股最近N个非停牌日进行计算
        """
        Turnover = self.df_sb["turnover_rate"]
        turnover20 = Turnover.rolling(20, 1).mean()
        return turnover20

    def VOL240(self):
        """
        因子描述：
        240日平均换手率（240-day Turnover Rate）。
        计算方法：
        最近240个交易日的平均换手率。
        """
        Turnover = self.df_sb["turnover_rate"]
        turnover240 = Turnover.rolling(240, 1).mean()
        return turnover240

    def VOL5(self):
        """
        因子描述：
        5日平均换手率（5-day Turnover Rate）。
        计算方法：
        最近5个交易日的平均换手率。
        注：取个股最近N个非停牌日进行计算
        """
        Turnover = self.df_sb["turnover_rate"]
        turnover5 = Turnover.rolling(5, 1).mean()
        return turnover5

    def VOL60(self):
        """
        因子描述：
        60日平均换手率（60-day Turnover Rate）。
        计算方法：
        最近60个交易日的平均换手率。
        """
        Turnover = self.df_sb["turnover_rate"]
        turnover60 = Turnover.rolling(60, 1).mean()
        return turnover60

    def _MA(self, N, any_series):
        MA = any_series.rolling(window=N, center=False).mean()
        return MA

    def VOSC(self):
        """
        因子描述：
        成交量震荡（Volume Oscillator），又称移动平均成交量指标，但它并非仅仅计算成交量的移动平均线，而是通过对成
        交量的长期移动平均线和短期移动平均线之间的比较，分析成交量的运行趋势和及时研判趋势转变方向。
        计算方法：
        VOSC = 100 * (MA(volume, 12) - MA(volume, 26)) / MA(volume, 12)
        """
        adj_volume = self.df_OHLC['volume']
        VOSC = 100 * (self._MA(12, adj_volume) - self._MA(26, adj_volume)) / self._MA(12, adj_volume)
        return VOSC

    def VR(self):
        """
        因子描述：
        成交量比率（Volume Ratio）。通过分析股价上升日成交额（或成交量，下同）与股价下降日成交额比值，从而掌握市
        场买卖气势的中期技术指标。
        计算方法：
        对任一交易日，若close > prev_close，当日成交量为AV，否则AV = 0。将N日内AV加和后记为AVS。
        对任一交易日，若close < prev_close，当日成交量为BV，否则BV = 0。将N日内BV加和后记为BVS。
        对任一交易日，若close = prev_close，当日成交量为CV，否则CV = 0。将N日内CV加和后记为CVS。
        VR = (AVS + CVS / 2) / (BVS + CVS / 2)。
        N取24，即24个交易日。
        """
        date_length = 24
        adj_volume = self.df_OHLC['volume']
        adj_close = self.df_OHLC['close']
        AV = BV = CV = adj_volume
        diff = adj_close.diff(1)
        AV = AV.where(diff > 0, 0)
        BV = BV.where(diff < 0, 0)
        CV = CV.where(diff == 0, 0)
        AVS = AV.rolling(window=date_length, center=False).sum()
        BVS = BV.rolling(window=date_length, center=False).sum()
        CVS = CV.rolling(window=date_length, center=False).sum()
        VR = (AVS + CVS / 2) / (BVS + CVS / 2)
        return VR

    def VROC12(self):
        """
        因子描述：
        12日量变动速率指标（12-day Volume Rate of Change）。以今天的成交量和N天前的成交量比较，通过计算某一段时
        间内成交量变动的幅度，应用成交量的移动比较来测量成交量运动趋向，达到事先探测成交量供需的强弱，进而分析成交
        量的发展趋势及其将来是否有转势的意愿，属于成交量的反趋向指标。
        计算方法：
        N取 12。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        volume = self.df_OHLC["volume"]
        lag_volume = volume.shift(12)
        VROC12 = (volume / lag_volume - 1) * 100
        return VROC12

    def VROC6(self):
        """
        因子描述：
        12日量变动速率指标（6-day Volume Rate of Change）。以今天的成交量和N天前的成交量比较，通过计算某一段时
        间内成交量变动的幅度，应用成交量的移动比较来测量成交量运动趋向，达到事先探测成交量供需的强弱，进而分析成交
        量的发展趋势及其将来是否有转势的意愿，属于成交量的反趋向指标。
        计算方法：
        N取 6。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        volume = self.df_OHLC["volume"]
        lag_volume = volume.shift(6)
        VROC6 = (volume / lag_volume - 1) * 100
        return VROC6

    def VSTD10(self):
        """
        因子描述：
        10日成交量标准差（10-day Volume Standard Deviation）。考察成交量的波动程度。
        计算方法：
        VSTD = std(volume, N)
        N取 10。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        volume = self.df_OHLC["volume"]
        VSTD10 = volume.rolling(10).std()
        return VSTD10

    def VSTD20(self):
        """
        因子描述：
        20日成交量标准差（20-day Volume Standard Deviation）。考察成交量的波动程度。
        计算方法：
        VSTD = std(volume, N)
        N取 20。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        volume = self.df_OHLC["volume"]
        VSTD20 = volume.rolling(20).std()
        return VSTD20

    def _TVMA(self, date_length):
        deal_amount = self.df_OHLC['amount']
        TVMA = self._MA(date_length, deal_amount) / 1e6
        return TVMA

    def TVMA20(self):
        """
        因子描述：
        20日成交金额的移动平均值（20-day Turnover Value Moving Average）。
        计算方法：
        TVMA = MA(value, N)/1e6
        N=20。
        注：取个股最近N个非停牌日进行计算
        """
        return self._TVMA(20)

    def TVMA6(self):
        """
        因子描述：
        6日成交金额的移动平均值（6-day Turnover Value Moving Average）。
        计算方法：
        TVMA = MA(value, N)/1e6
        N=6。
        注：取个股最近N个非停牌日进行计算
        """
        return self._TVMA(6)

    def TVSTD20(self):
        """
        因子描述：
        20日成交金额的标准差（20-day Turnover Value Standard Deviation）。
        计算方法：
        TVSTD = std(value, N)/1e6
        N=20。
        注：取个股最近N个非停牌日进行计算
        """
        value = self.df_OHLC["amount"]
        TVSTD20 = value.rolling(20).std() / 1e6
        return TVSTD20

    def TVSTD6(self):
        """
        因子描述：
        20日成交金额的标准差（20-day Turnover Value Standard Deviation）。
        计算方法：
        TVSTD = std(value, N)/1e6
        N=6
        """
        value = self.df_OHLC["amount"]
        TVSTD6 = value.rolling(6).std() / 1e6
        return TVSTD6

    def STOM(self):
        """
        因子描述：
        月度换手率对数。
        计算方法：
        每个月按照21个交易日计算。
        """
        turnover = self.df_sb["turnover_rate"]
        TOM = turnover.rolling(21).sum()
        STOM = np.log(TOM)
        return STOM

    def STOQ(self):
        """
        因子描述：
        3个月换手率对数平均。
        计算方法：
        其中，T=3个月，这里每个月按照21个交易日计算。
        """
        turnover = self.df_sb["turnover_rate"]
        TOQ = turnover.rolling(63).sum() / 3
        STOQ = np.log(TOQ)
        return STOQ

    def STOA(self):
        """
        因子描述：
        12个月换手率对数平均。
        计算方法：
        其中，T=12个月，每个月按照21个交易日计算。
        """
        turnover = self.df_sb["turnover_rate"]
        TOA = turnover.rolling(252).sum() / 12
        STOA = np.log(TOA)
        return STOA
