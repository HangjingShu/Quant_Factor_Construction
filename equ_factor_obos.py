from pyfinance import ols
from .base_FG_Equ import Equ
import numpy as np
import pandas as pd
"""
get series from df_bic
calculate if necessary
drop duplicated records
expand the index list of df_bic by df_sb
div df_sb if necessary
"""


# 股票因子_超买超卖型 62
class EquFactorObos(Equ):
    def __init__(self, OHLC, Index_OHLC, sb, if_qfq=False):
        super().__init__()
        self.OHLC = OHLC
        self.sb = sb
        self.Index_OHLC = Index_OHLC
        self.if_qfq = if_qfq

    def _get_DTM(self):
        """
            若当日开盘价大于昨日开盘价，则DTM = max(highest – open, open – prev_open)，否则DTM = 0。
        """
        adj_high = self.df_OHLC['high']
        adj_open = self.df_OHLC['open']
        temp1 = adj_high - adj_open
        temp2 = adj_open - adj_open.shift(1)
        DTM = temp1.mask(temp2 > temp1, temp2)
        DTM = DTM.mask(temp2 <= 0, 0)
        return DTM

    def _get_DBM(self):
        """
            若当日开盘价小于昨日开盘价，则DBM = max(open – lowest, prev_open - open)，否则DBM=0。
        """
        adj_open = self.df_OHLC['open']
        adj_low = self.df_OHLC['low']
        temp2 = adj_open - adj_open.shift(1)
        temp3 = adj_open - adj_low
        DBM = temp3.mask(temp2 > temp3, temp2)
        DBM = DBM.mask(temp2 >= 0, 0)
        return DBM


    def ADTM(self):
        """
        因子描述：
        动态买卖气指标 (Moving dynamic indicator)，用开盘价的向上波动幅度和向下波动幅度的距离差值来描述人气高低的指标。
        计算方法：
        1. 若当日开盘价大于昨日开盘价，则DTM = max(highest – open, open – prev_open)，否则DTM=0。
        2. 若当日开盘价小于昨日开盘价，则DBM = max(open – lowest, prev_open - open)，否则DBM=0。
        3. STM为N日内DTM之和，SBM为N日内DBM之和。N =23。
        4. ADTM = (STM - SBM) / max(STM, SBM)。
        """
        STM = self.STM()
        SBM = self.SBM()
        ADTM = (STM-SBM)/STM.mask(SBM > STM, SBM)
        return ADTM

    def _MA(self, N, any_series):
        MA = any_series.rolling(window=N, center=False).mean()
        return MA

    def ATR14(self):
        """
        因子描述：
        N日均幅指标（14-day Average True Range）。取一定时间周期内的股价波动幅度的移动平均值，是显示市场变化率的指标，主要用于研判买卖时机。
        计算方法：
        1. TR = max(highest-lowest, abs(highest-prev_close), abs(lowest-prev_close))
        2. 上市后，前N日内的ATR等于N日TR平均，自N+1日起使用如下迭代公式计算：

        N = 14。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        N = 14
        adj_high = self.df_OHLC['high']
        adj_close = self.df_OHLC['close']
        adj_low = self.df_OHLC['low']
        temp1 = adj_high - adj_low
        temp2 = (adj_high - adj_close.shift(1)).abs()
        temp3 = (adj_low - adj_close.shift(1)).abs()
        TR = temp1.mask(temp2 > temp1, temp2)
        TR = TR.mask(temp3 > TR, temp3)
        MA = self._MA(N, TR)
        TR.iloc[0:N] = MA.iloc[0:N]
        ATR = TR.ewm(alpha=1 / N, adjust=False, min_periods=1).mean()
        return ATR

    def ATR6(self):
        """
        因子描述：
        6日均幅指标（6-day Average True Range。取一定时间周期内的股价波动幅度的移动平均值，是显示市场变化率的指标，主要用于研判买卖时机。
        计算方法：
        1. TR = max(highest-lowest, abs(highest-prev_close), abs(lowest-prev_close))。
        2. 上市后，前N日内的ATR等于N日TR平均，自N+1日起使用如下迭代公式计算：

        N=6
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        N = 6
        adj_high = self.df_OHLC['high']
        adj_close = self.df_OHLC['close']
        adj_low = self.df_OHLC['low']
        temp1 = adj_high - adj_low
        temp2 = (adj_high - adj_close.shift(1)).abs()
        temp3 = (adj_low - adj_close.shift(1)).abs()
        TR = temp1.mask(temp2 > temp1, temp2)
        TR = TR.mask(temp3 > TR, temp3)
        MA = self._MA(N, TR)
        TR.iloc[0:N] = MA.iloc[0:N]
        ATR = TR.ewm(alpha=1 / N, adjust=False, min_periods=1).mean()
        return ATR

    def _BIAS(self, N):
        """
        因子描述：
        乖离率 (N-day Bias Ratio/BIAS)，简称Y值，是移动平均原理派生的一项技术指标，表示股价偏离趋向指标斩百分比值。
        计算方法：
        BIAS = (收盘价 / MA(收盘价, N) - 1) * 100
        N取10
        """
        adj_close = self.df_OHLC['close']
        BIAS = (adj_close / self._MA(N, adj_close) - 1) * 100
        return BIAS

    def BIAS10(self):
        """
        因子描述：
        乖离率 (10-day Bias Ratio/BIAS)，简称Y值，是移动平均原理派生的一项技术指标，表示股价偏离趋向指标斩百分比值。
        计算方法：
        BIAS = (收盘价 / MA(收盘价, N) - 1) * 100
        N取10
        """
        return self._BIAS(10)

    def BIAS20(self):
        """
        因子描述：
        乖离率 (20-day Bias Ratio/BIAS)，简称Y值，是移动平均原理派生的一项技术指标，表示股价偏离趋向指标斩百分比值。
        计算方法：
        BIAS = (收盘价 / MA(收盘价, N) - 1) * 100
        N取20
        """
        return self._BIAS(20)

    def BIAS5(self):
        """
        因子描述：
        乖离率 (5-day Bias Ratio/BIAS)，简称Y值，是移动平均原理派生的一项技术指标，表示股价偏离趋向指标斩百分比值。
        计算方法：
        BIAS = (收盘价 / MA(收盘价, N) - 1) * 100
        N取5
        """
        return self._BIAS(5)

    def BIAS60(self):
        """
        因子描述：
        乖离率 (60-day Bias Ratio/BIAS)，简称Y值，是移动平均原理派生的一项技术指标，表示股价偏离趋向指标斩百分比值。
        计算方法：
        BIAS = (收盘价 / MA(收盘价, N) - 1) * 100
        N取60
        """
        return self._BIAS(60)

    def BollDown(self):
        """
        因子描述：
        下轨线（布林线）指标（Lower Bollinger Bands），它是研判股价运动趋势的一种中长期技术分析工具。
        计算方法：
        中轨线为N日的移动平均线，下轨线为中轨线-两倍标准差。计算取N=20。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        MA = self._MA(20,adj_close)
        BollDown = MA - adj_close.rolling(window=20, center=False).std()
        return BollDown

    def BollUp(self):
        """
        因子描述：
        上轨线（布林线）指标（Upper Bollinger Bands），它是研判股价运动趋势的一种中长期技术分析工具。
        计算方法：
        中轨线为N日的移动平均线；上轨线为中轨线+两倍标准差。计算取N=20。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        MA = self._MA(20, adj_close)
        BollUp = MA + adj_close.rolling(window=20, center=False).std()
        return BollUp

    def _CCI(self, N):
        """
        因子描述：
        10日顺势指标（10-day Commodity Channel Index），专门测量股价是否已超出常态分布范围。CCI指标波动于正无穷大到负无穷大之间，不会出现指标钝化现象，有利于投资者更好地研判行情，特别是那些短期内暴
        涨暴跌的非常态行情。
        计算方法：
        1. 计算典型价格：TYP = (close + highest + lowest) / 3。
        2. 计算典型价格的移动平均：MATYP = MA(TYP, N)。
        3. 计算典型价格的偏差：

        4. CCI = (TYP - TYPMA) / 0.015 / DEV
        """
        adj_close = self.df_OHLC['close']
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        TYP = (adj_close + adj_high + adj_low) / 3
        MATYP = self._MA(N, TYP)
        DEV = ((TYP - MATYP).abs()).rolling(window=N, center=False).sum()
        CCI = (TYP - MATYP) / 0.015 / DEV
        return CCI



    def CCI10(self):
        """
        因子描述：
        10日顺势指标（10-day Commodity Channel Index），专门测量股价是否已超出常态分布范围。CCI指标波动于正无穷大到负无穷大之间，不会出现指标钝化现象，有利于投资者更好地研判行情，特别是那些短期内暴
        涨暴跌的非常态行情。
        计算方法：
        1. 计算典型价格：TYP = (close + highest + lowest) / 3。
        2. 计算典型价格的移动平均：MATYP = MA(TYP, N)。
        3. 计算典型价格的偏差：

        4. CCI = (TYP - TYPMA) / 0.015 / DEV
        N取10。
        """
        return self._CCI(10)

    def CCI20(self):
        """
        因子描述：
        20日顺势指标（20-day Commodity Channel Index），专门测量股价是否已超出常态分布范围。CCI指标波动于正无穷大到负无穷大之间，不会出现指标钝化现象，有利于投资者更好地研判行情，特别是那些短期内暴
        涨暴跌的非常态行情。
        计算方法：
        1. 计算典型价格：TYP = (close + highest + lowest) / 3。
        2. 计算典型价格的移动平均：MATYP = MA(TYP, N)。
        3. 计算典型价格的偏差：

        4. CCI = (TYP - TYPMA) / 0.015 / DEV
        N取20
        """
        return self._CCI(20)

    def CCI5(self):
        """
        因子描述：
        5日顺势指标（5-day Commodity Channel Index），专门测量股价是否已超出常态分布范围。CCI指标波动于正无穷大到负无穷大之间，不会出现指标钝化现象，有利于投资者更好地研判行情，特别是那些短期内暴涨暴
        跌的非常态行情。
        计算方法：
        1. 计算典型价格：TYP = (close + highest + lowest) / 3。
        2. 计算典型价格的移动平均：MATYP = MA(TYP, N)。
        3. 计算典型价格的偏差：

        4. CCI = (TYP - TYPMA) / 0.015 / DEV
        N取5
        """
        return self._CCI(5)

    def CCI88(self):
        """
        因子描述：
        88日顺势指标（88-day Commodity Channel Index），专门测量股价是否已超出常态分布范围。CCI指标波动于正无穷大到负无穷大之间，不会出现指标钝化现象，有利于投资者更好地研判行情，特别是那些短期内暴
        涨暴跌的非常态行情。
        计算方法：
        1. 计算典型价格：TYP = (close + highest + lowest) / 3。
        2. 计算典型价格的移动平均：MATYP = MA(TYP, N)。
        3. 计算典型价格的偏差：

        4. CCI = (TYP - TYPMA) / 0.015 / DEV
        N取88
        """
        return self._CCI(88)

    def _RSV(self, N=9):
        """
        .计算N日内的未成熟随机值RSV（Raw Stochastic Value）：
        RSV = (close – lowest) / (highest – lowest) * 100
        """
        adj_close = self.df_OHLC['close']
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        temp1 = adj_low.rolling(window=N, center=False).min()
        lowest = temp1.where(temp1.notnull(), adj_low)
        temp2 = adj_high.rolling(window=N, center=False).max()
        highest = temp2.where(temp2.notnull(), adj_high)
        RSV = ((adj_close - lowest) / (highest - lowest)) * 100
        return RSV

    def KDJ_K(self):
        """
        因子描述：
        随机指标 (K Stochastic Oscillator)。它综合了动量观念、强弱指标及移动平均线的优点，用来度量股价脱离价格正常范围的变异程度。
        计算方法：
        1. 计算N日内的未成熟随机值RSV（Raw Stochastic Value）：
        RSV = (close – lowest) / (highest – lowest) * 100
        其中N = 9。
        2. 当日K值=（2 * 前日K值 + 当日RSV）/ 3，若前日K值缺失使用50代替。
        未解决缺失用50替代问题
        """
        RSV = self._RSV().fillna(50)
        KDJ_K = RSV.ewm(com=2, adjust=False).mean()
        return KDJ_K

    def KDJ_D(self):
        """
        因子描述：
        随机指标 (D Stochastic Oscillator)。它综合了动量观念、强弱指标及移动平均线的优点，用来度量股价脱离价格正常范围的变异程度。
        计算方法：
        1. 计算N日内的未成熟随机值RSV（Raw Stochastic Value）：
        RSV = (close – lowest) / (highest – lowest) * 100
        其中N = 9。
        2. 当日D值=（2 * 前日D值 + 当日K值）/ 3，若前日D值缺失使用50代替。其中K值指KDJ_K的因子值。
        """
        RSV = self._RSV().fillna(50)
        KDJ_K = RSV.ewm(com=2, adjust=False).mean()
        KDJ_D = KDJ_K.ewm(com=2, adjust=False).mean()
        return KDJ_D

    def KDJ_J(self):
        """
        因子描述：
        随机指标 (J Stochastic Oscillator)。它综合了动量观念、强弱指标及移动平均线的优点，用来度量股价脱离价格正常范围的变异程度。
        计算方法：
        1. 计算N日内的未成熟随机值RSV（Raw Stochastic Value）：
        RSV = (close – lowest) / (highest – lowest) * 100
        其中N = 9。
        2. 当日J值 = 3 * 当日K值 – 2 * 当日D值。其中K值指KDJ_K的因子值,D值指KDJ_D的因子值。
        """
        RSV = self._RSV().fillna(50)
        KDJ_K = RSV.ewm(com=2, adjust=False).mean()
        KDJ_D = KDJ_K.ewm(com=2, adjust=False).mean()
        KDJ_J = 3 * KDJ_K - 2 * KDJ_D
        return KDJ_J

    def _ROC(self,N):
        adj_close = self.df_OHLC['close']
        ROC = 100 * (adj_close / adj_close.shift(N) - 1)
        return ROC

    def ROC6(self):
        """
        因子描述：
        6日变动速率（6-day Price Rate of Change）。是一个动能指标，其以当日的收盘价和N天前的收盘价比较，通过计算股价某一段时间内收盘价变动的比例，应用价格的移动比较来测量价位动量，属于反趋向的指标之一。
        计算方法：

        N取6日
        """
        return self._ROC(6)

    def ROC20(self):
        """
        因子描述：
        20日变动速率（20-day Price Rate of Change）。是一个动能指标，其以当日的收盘价和N天前的收盘价比较，通过计算股价某一段时间内收盘价变动的比例，应用价格的移动比较来测量价位动量，属于反趋向的指标之一。
        计算方法：

        N取20日
        """
        return self._ROC(20)

    def SBM(self, N=23):
        """
        因子描述：
        计算ADTM因子的中间变量 (mediator in calculating ADTM)。
        计算方法：
        1. 若当日开盘价小于昨日开盘价，则DBM = max(open – lowest, prev_open - open)，否则DBM=0。
        2. SBM为N日内DBM之和。N =23。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        DBM = self._get_DBM()
        SBM = DBM.rolling(window=N, center=False).sum()
        return SBM

    def STM(self, N=23):
        """
        因子描述：
        计算ADTM因子的中间变量 (mediator in calculating ADTM)。
        计算方法：
        1. 若当日开盘价大于昨日开盘价，则DTM = max(highest – open, open- prev_open)，否则DTM=0。
        2. STM为N日内DTM之和。N =23。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        DTM = self._get_DTM()
        STM = DTM.rolling(window=N, center=False).sum()
        return STM

    def _EMA(self, N, date_length, adj_close):
        if date_length == 1:
            return adj_close
        else:
            alpha = 2 / (N + 1)
            EMA = alpha * adj_close + (1 - alpha) * self._EMA(N, date_length-1, adj_close.shift(1))
            return EMA

    def UpRVI(self):
        """
        因子描述：
        计算RVI因子的中间变量 (Up relative volatility factor)。
        计算方法：
        1. 计算每日收盘价标准差SD，Dorsey 建议向前取10个交易日区间计算。
        2. 若当日收盘价大于昨日收盘价，当日记为上升日，USD = SD，DSD = 0；若当日收盘价小于昨日收盘价，当日记为下降日，USD = 0，DSD = SD。
        3. 对过去一段时间内的上升日和下降日的标准差求N日Wilder’s Smoothing，Dorsey 建议向前取14个交易日（计算2N-1日的EMA将得到相同结果，但是速度更快）。
        UpRVI = EMA(USD, 2N-1)
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        N = 14
        adj_close = self.df_OHLC['close']
        SD = adj_close.rolling(window=10, center=False).std()
        temp = (adj_close - adj_close.shift(1)).where(SD.notnull(), np.nan)
        USD = SD.mask(temp < 0, 0)
        UpRVI = self._EMA(2 * N - 1, 2 * N - 1, USD)
        return UpRVI

    def DownRVI(self):
        """
        因子描述：
        计算RVI因子的中间变量 (Down relative volatility factor)。
        计算方法：
        1. 计算每日收盘价标准差SD，Dorsey 建议向前取10个交易日区间计算。
        2. 若当日收盘价大于昨日收盘价，当日记为上升日，USD = SD，DSD = 0；若当日收盘价小于昨日收盘价，当日记为下降日，USD = 0，DSD = SD。
        3. 对过去一段时间内的上升日和下降日的标准差求N日Wilder’s Smoothing，Dorsey 建议向前取14个交易日（计算2N-1日的EMA将得到相同结果，但是速度更快）。
        DownRVI = EMA(DSD, 2N-1)
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        N = 14
        adj_close = self.df_OHLC['close']
        SD = adj_close.rolling(window=10, center=False).std()
        temp = (adj_close - adj_close.shift(1)).where(SD.notnull(), np.nan)
        DSD = SD.mask(temp > 0, 0)
        DownRVI = self._EMA(2 * N - 1, 2 * N - 1, DSD)
        return DownRVI

    def RVI(self):
        """
        因子描述：
        相对离散指数（Relative Volatility Index）。又称“相对波动性指标”，用于测量价格的发散趋势，其原理与相对强弱指标（RSI）类似，但它是以价格的方差而不是简单的升跌来测量价格变化的强度。主要用作辅助的确认指标，即配合均线系统、动量指标或其它趋势指标使用。
        计算方法：
        1. 计算每日收盘价标准差SD，Dorsey 建议向前取10个交易日区间计算。
        2. 若当日收盘价大于昨日收盘价，当日记为上升日，USD = SD，DSD = 0；若当日收盘价小于昨日收盘价，当日记为下降日，USD = 0，DSD = SD。
        3. 对过去一段时间内的上升日和下降日的标准差求N日Wilder’s Smoothing，Dorsey 建议向前取14个交易日（计算2N-1日的EMA将得到相同结果，但是速度更快）。
        4. UpRVI = EMA(USD, 2N-1)，DownRVI = EMA(DSD, 2N-1)；
        5. RVI = 100 * UpRVI / (UpRVI + DownRVI)。
        """
        RVI = 100 * self.UpRVI() / (self.UpRVI() + self.DownRVI())
        return RVI

    def SRMI(self):
        """
        因子描述：
        修正动量指标 (Modified Momentom Index)。
        计算方法：
        在当日收盘价小于前一交易日时，以前一交易日作为衡量基准；当日收盘价大于前一交易日时，以当日作为衡量基准。N = 10。
        """
        adj_close = self.df_OHLC['close']
        temp1 = adj_close.shift(10)
        temp2 = adj_close.mask(temp1 > adj_close, temp1)
        SRMI = (adj_close - temp1) / temp2
        return SRMI

    def RSI(self):
        """
        因子描述：
        相对强弱指标（Relative Strength Index），通过比较一段时期内的平均收盘涨数和平均收盘跌数来分析市场买沽盘的意向和实力，据此预测趋势的持续或者转向。
        计算方法：
        1. 若当日收盘价高于前日收盘价，定义当日上涨指标U_i = close – prev_close，下跌指标D_i = 0。
        2. 若当日收盘价低于前日收盘价，定义当日上涨指标U_i = 0，下跌指标D_i = prev_close – close。
        3. 若当日收盘价与前日收盘价持平，定义上涨和下跌指标U_i = D_i = 0。
        4. 相对强度RS = MA(U, N) / MA(D, N)。
        5. RSI = 100 – 100 / (1 + RS)。
        N取12。
        """
        adj_close = self.df_OHLC['close']
        U_i = adj_close.diff(1)
        U_i = U_i.mask(U_i <= 0, 0)
        D_i = adj_close.shift(1) - adj_close
        D_i = D_i.mask(D_i <= 0, 0)
        RS = self._MA(12, U_i) / self._MA(12, D_i)
        RSI = 100 - 100 / (1 + RS)
        return RSI

    def ChandeSD(self):
        """
        因子描述：
        计算CMO因子的中间变量 (mediator in calculating CMO)。SD是今日收盘价与昨日收盘价（下跌日）差值的绝对值加总。若当日上涨，则增加值为0。
        计算方法：
        N取20。

        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        temp1 = adj_close.shift(1) - adj_close
        temp2 = temp1.mask(temp1 < 0, 0)
        SD = temp2.rolling(window=20, center=False).sum()
        return SD

    def ChandeSU(self):
        """
        因子描述：
        计算CMO因子的中间变量 (mediator in calculating CMO)。SU是今日收盘价与昨日收盘价（上涨日）差值加总。若当日下跌，则增加值为0。
        计算方法：

        N取20。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        temp1 = adj_close - adj_close.shift(1)
        temp2 = temp1.mask(temp1 < 0, 0)
        SU = temp2.rolling(window=20, center=False).sum()
        return SU

    def CMO(self):
        """
        因子描述：
        钱德动量摆动指标（Chande Momentum Osciliator）。由Tushar Chande发明，与其他动量指标摆动指标如相对强弱指标（RSI）和随机指标（KDJ）不同，钱德动量指标在计算公式的分子中采用上涨日和下跌日的数据。
        计算方法：
        1. SU是今日收盘价与昨日收盘价（上涨日）差值加总。若当日下跌，则增加值为0；
        2. SD是今日收盘价与昨日收盘价（下跌日）差值的绝对值加总。若当日上涨，则增加值为0。
        3. CMO = (SU - SD) / (SU + SD) * 100。
        N取20。
        """
        SU = self.ChandeSU()
        SD = self.ChandeSD()
        CMO = (SU - SD) / (SU + SD) * 100
        return CMO

    def DBCD(self):
        """
        因子描述：
        异同离差乖离率 (Bias convergence divergence)。先计算乖离率BIAS，然后计算不同日的乖离率之间的离差，最后对离差进行指数移动平滑处理。优点是能够保持指标的紧密同步，并且线条光滑，信号明确，能够有效的过滤掉伪信号。
        计算方法：
        1. BIAS = (close / MA(close, N) - 1) * 100
        2. DIF = BIAS(t) – BIAS(t - M)
        3. DBCD = EMA(DIF, T)
        取N = 5, M = 16, T = 17。
        """
        adj_close = self.df_OHLC['close']
        BIAS = (adj_close / self._MA(5, adj_close) - 1) * 100
        DIF = BIAS.diff(16)
        DBCD = self._EMA(17, 17, DIF)
        return DBCD

    def ARC(self):
        """
        因子描述：
        变化率指数均值 (Average Rate of Change)。股票的价格变化率RC指标的均值，用以判断前一段交易周期内股票的平均价格变化率。
        计算方法：
        其中N=50，1/N为指数移动平均的加权系数 。
        """
        adj_close = self.df_OHLC['close']
        def get_EMA_ARC(N, date_length, adj_close):
            if date_length == 1:
                return adj_close
            else:
                EMA = 1 / N * adj_close + (1 - 1 / N) * get_EMA_ARC(N, date_length - 1,adj_close.shift(1))
                return EMA
        RC = adj_close / adj_close.shift(50)
        ARC = get_EMA_ARC(50, 50, RC)
        return ARC

    def HBETA(self):
        """
        因子描述：
        历史贝塔（Historical daily beta），过往12个月中，个股日收益对市场组合日收益、自身三阶收益率进行回归，市场组合日收益的系数。
        计算方法：
        其中市场组合日收益 的计算采用沪深300的数据。回归结果中的 即为历史贝塔HBETA。
        表示日收益，CloseIndex表示指数今收盘，PreCloseIndex表示指数昨收盘。
        使用12个自然月数据进行计算，缺失值大于90天，则因子值为nan。
        """
        pass
        x = pd.DataFrame()
        x['r2'] = self.df_OHLC['pct_chg'].shift(1)
        x['r3'] = self.df_OHLC['pct_chg'].shift(2)
        x['r4'] = self.df_OHLC['pct_chg'].shift(3)
        x['rm'] = self.df_Index_OHLC['pct_chg']
        y = self.df_OHLC['pct_chg']
        roll = ols.PandasRollingOLS(y=y, x=x, window=250)
        return roll.beta['rm']




    def HSIGMA(self):
        """
        因子描述：
        历史波动（Historical daily sigma），过往12个月中，个股日收益对市场组合日收益、自身三阶收益率进行回归，得到的残差的方差。
        计算方法：
        其中市场组合日收益 的计算采用沪深300 的数据
        说明：
        p=4, n为序列的长度
        表示日收益，CloseIndex表示指数今收盘，PreCloseIndex表示指数昨收盘。
        使用12个自然月数据进行计算，缺失值大于90天，则因子值为nan。
        """
        N = 250
        x = pd.DataFrame()
        x['r2'] = self.df_OHLC['pct_chg'].shift(1)
        x['r3'] = self.df_OHLC['pct_chg'].shift(2)
        x['r4'] = self.df_OHLC['pct_chg'].shift(3)
        x['rm'] = self.df_Index_OHLC['pct_chg']
        y = self.df_OHLC['pct_chg']
        roll = ols.PandasRollingOLS(y=y, x=x, window=N)
        ttt = roll.resids.sum(level=0)
        HSIGMA = 1/(N - 4 - 1) * ttt
        return HSIGMA


    def DDNSR(self):
        """
        因子描述：
        下跌波动（Downside standard deviations ratio），过往12个月中，市场组合日收益为负时，个股日收益标准差和市场组合日收益标准差之比。
        计算方法：
        说明：
        市场组合日收益 的计算采用沪深300 的数据，仅考虑市场回报为负的数据。
        r表示个股日收益，CloseIndex表示指数今收盘，PreCloseIndex表示指数昨收盘。
        说明：在过去12个月，市场回报为负的日期中，如果个股日收益有效数据长度小于20，则因子值为nan。
        """
        r = self.df_OHLC['pct_chg']
        rm = self.df_Index_OHLC['pct_chg']
        df = pd.DataFrame(r)
        nrm = rm.mask(rm >= 0, np.nan)
        df['r_index'] = rm
        df[df['r_index'] >= 0] = np.nan
        # del r["r_index"]
        sd_r = df['pct_chg'].rolling(window=240, min_periods=20).std()
        sd_nrm = nrm.rolling(window=240, min_periods=1).std()
        DDNSR = sd_r / sd_nrm
        return DDNSR

    def DDNCR(self):
        """
        因子描述：
        下跌相关系数（Downside correlation），过往12个月中，市场组合日收益为负时，个股日收益关于市场组合日收益的相关系数。
        计算方法：
        说明：
        市场组合日收益 的计算采用沪深300 的数据，仅考虑市场回报为负的数据。
        r表示个股日收益，CloseIndex表示指数今收盘，PreCloseIndex表示指数昨收盘。
        说明：在过去12个月，市场回报为负的日期中，如果个股日收益有效数据长度小于20，则因子值为nan。
        """
        r = self.df_OHLC['pct_chg']
        rm = self.df_Index_OHLC['pct_chg']
        df = pd.DataFrame(r)
        nrm = rm.mask(rm >= 0, np.nan)
        df['r_index'] = rm
        df[df['r_index'] >= 0] = np.nan
        DDNCR = df['pct_chg'].rolling(window=240, min_periods=20).corr(other=nrm, pairwise=True)
        return DDNCR

    def BackwardADJ(self):
        """
        因子描述：
        股价向后复权因子（Fowward dividend adjusted price factor ）。复权是对股价和成交量进行权息修复，按照股票的实际涨跌绘制股价走势图，并把成交量调整为相同的股本口径。向后复权是指保持基准日期（上市日期）开始
        的股价不变，将以后的股价进行复权调整。在每个交易日，计算除权因子：将自基准日期以来的所有除权因子累乘，得到对应每个交易日的除权因子。
        计算方法：
        其中ClosePrice表示除权日收盘价，AllotPrice表示配股价，AllorRatio表示配股比率，CashDivPerShare表示每股派息，DivRatio表示送股比率，TransRatio表示每股转增股比率。
        """
        adj_factor = self.df_OHLC['adj_factor']
        return adj_factor

    def _REVS(self,N):
        adj_close = self.df_OHLC['close']
        REVS = adj_close / adj_close.shift(N)
        return REVS

    def REVS5(self):
        """
        因子描述：
        过去5天的价格动量。
        计算方法：

        close为收盘价。
        注1：若公司在过去的5天内有停牌，停牌日也计算在统计天数内;
        注2：若公司在今天停牌，不计算该因子的值；下同。
        """
        return self._REVS(5)

    def REVS10(self):
        """
        因子描述：
        过去10天的价格动量。
        计算方法：

        close为收盘价。
        注1：若公司在过去的10天内有停牌，停牌日也计算在统计天数内;
        注2：若公司在今天停牌，不计算该因子的值；下同。
        """
        return self._REVS(10)

    def REVS20(self):
        """
        因子描述：
        过去20天的价格动量。
        计算方法：

        close为收盘价。
        注1：若公司在过去的20天内有停牌，停牌日也计算在统计天数内;
        注2：若公司在今天停牌，不计算该因子的值；下同。
        """
        return self._REVS(20)


    def DVRAT(self):
        """
        因子描述：
        收益相对波动（Daily returns variance ratio-serial dependence in daily returns）。
        计算方法：
        记
        为第i支股票的日收益，
        为每日的无风险收益，则该股票当日的超额日收益
        收益相对波动可表示为:
        T 为过往24 个月中的交易日数，q = 10。
        代码计算中将每日无风险收益
        按0 处理，最终结果舍去了交易日不足一半时间的结果。
        """

        pct_chg = self.df_OHLC['pct_chg'].fillna(method='ffill')
        r_f = 0
        r = (pct_chg - r_f)
        sigma = (np.square(r)).rolling(window=280, center=False).sum() / 479
        m = 10 * (240 - 10 + 1) * (1 - 10 / 240)
        temp1 = r.rolling(window=10, center=False).sum()
        temp2 = np.square(temp1)
        temp3 = temp2.rolling(window=240 - 10 + 1, center=False).sum()
        sigmaq = temp3 / m
        DVRAT = sigmaq / sigma - 1
        return DVRAT

    def DDNBT(self):
        """
        因子描述：
        下跌贝塔（Downside beta），过往12 个月中，市场组合日收益为负时，个股日收益关于市场组合日收益的回归系数。
        计算方法：

        其中市场组合日收益 的计算采用沪深300 的数据，仅考虑市场回报为负的数据
        回归结果中的 即为下跌贝塔DDNBT；
         表示个股日收益，CloseIndex表示指数今收盘，PreCloseIndex表示指数昨收盘。
        说明：在过去12个月，市场回报为负的日期中，如果个股日收益有效数据长度小于20，则因子值为nan。
        """
        r = self.df_OHLC['pct_chg']
        rm = self.df_Index_OHLC['pct_chg']
        nrm = rm.mask(rm >= 0, np.nan)
        corrcoef = r.rolling(window=240, center=False, min_periods=1).corr(other=nrm, pairwise=True)
        std_x = nrm.rolling(window=240, center=False, min_periods=1).std()
        beta = (corrcoef * r.rolling(window=240, center=False).std()).div(std_x.squeeze(), axis=0)
        return beta

    def Skewness(self):
        """
        因子描述：
        股价偏度（Skewness of price during the last 20 days），过去20 个交易日股价的偏度。
        计算方法：

        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        skew = adj_close.rolling(window=20, center=False).skew()
        return skew

    def FiftyTwoWeekHigh(self):
        """
        因子描述：
        当前价格处于过去1 年股价的位置（Price level during the pasted 240 trade days）。
        计算方法：

        说明：
        此处定义一年为240个交易日
        """
        adj_close = self.df_OHLC['close']
        min = adj_close.rolling(window=240, center=False).min()
        max = adj_close.rolling(window=240, center=False).max()
        ftwh = (adj_close - min) / (max - min)
        return ftwh

    def CMRA(self):
        """
        因子描述：
        24 月累计收益（Monthly cumulative return range over the past 24 months）。
        计算方法：
        1. 定义过去T个月的累计收益率, T=1,2, 3, .... N
        1. 计算因子值
        其中 为Z(T)的最大值， 为Z(T)的最小值
        说明：
        N = 24，无风险收益率 , 一个月为一个完整自然月
        """
        n = 21
        adj_close = self.df_OHLC['close']
        r = pd.DataFrame()
        for i in range(0, -24, -1):
            r_tmp = (adj_close.shift(abs(i)*n)/adj_close.shift((abs(i)+1)*n)).apply(np.log)
            if i == 0:
                r[abs(i)] = r_tmp
            else:
                r[abs(i)] = r[abs(i)-1] + r_tmp
        r_max = r.max(axis=1)
        r_min = r.min(axis=1)
        CMRA = r_max - r_min
        return CMRA



    def ILLIQUIDITY(self):
        """
        因子描述：
        收益相对金额比（Daily return to turnover value during the last 20 days），过去20 个交易日收益相对成交金额的比例，结果乘以10亿。
        计算方法：
        符号说明：
        符号 描述 计算方法
        r 日收益 Latest
        Value 成交金额 Latest
        注：分母<0时，因子值为空
        """
        R = self.df_OHLC['pct_chg']
        deal_amount = self.df_OHLC['amount']
        Illiquidity = R.rolling(window=20, center=False).sum() / deal_amount.rolling(window=20,
                                                                                     center=False).sum() * 1e9
        return Illiquidity

    def Volatility(self):
        """
        因子描述：
        换手率相对波动率（Volatility of daily turnover during the last 20 days）。
        计算方法：

        其中TORate为换手率
        注：分母<0时，因子值为空
        """
        turnover_rate = self.df_sb['turnover_rate']
        sd=turnover_rate.rolling(window=20,center=False).std()
        mean=turnover_rate.rolling(window=20,center=False).mean()
        volatility=sd/mean
        return volatility

    def MFI(self):
        """
        因子描述：
        资金流量指标（Money Flow Index），该指标是通过反映股价变动的四个元素：上涨的天数、下跌的天数、成交量增加幅度、成交量减少幅度来研判量能的趋势，预测市场供求关系和买卖力道。
        计算方法：
        1. 计算典型价格：TYP = (close + highest + lowest) / 3。
        2. 当日现金流为典型价格和当日成交量的乘积：MF = TYP * volume。
        3. 若当日典型价格高于前日典型价格，定义当日现金流为正；若当日典型价格低于前日典型价格，定义当日现金流为负；若当日典型价格等于前日典型价格，当日被舍去。资金比率（Money Ratio）可如下计算：
        4. MFI = 100 – 100 / (1 + MR)
        N取14。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        adj_volume = self.df_OHLC['volume']
        TYP = (adj_close + adj_high + adj_low) / 3
        MF = TYP * adj_volume
        temp = TYP - TYP.shift(1)
        temp_1 = MF.where(temp > 0, 0)
        temp_2 = MF.where(temp < 0, 0)
        MR = temp_1.rolling(window=14, center=False).sum() / temp_2.rolling(window=14, center=False).sum()
        MFI = 100 - 100 / (1 + MR)
        return MFI

    def WVAD(self):
        """
        因子描述：
        威廉变异离散量（William's variable accumulation distribution），是一种将成交量加权的量价指标。用于测量从开盘价至收盘价期间，买卖双方各自爆发力的程度。
        计算方法：

        其中N = 24, 最后入库数值除以1e6。
        特别说明：上市不足30天的股票，值为空。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        adj_open = self.df_OHLC['open']
        adj_close = self.df_OHLC['close']
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        adj_volume = self.df_OHLC['volume']
        temp = (adj_close - adj_open) / (adj_high - adj_low) * adj_volume
        WVAD = temp.rolling(window=24, center=False).sum()
        WVAD = WVAD / 1e6
        return WVAD

    def MAWVAD(self):
        """
        因子描述：
        因子WVAD的均值 (Moving average of William’s variable accumulation distribution。
        计算方法：

        其中N = 24，M = 6, 最后入库数值除以1e6。
        特别说明：上市不足30天的股票，值为空。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        WVAD = self.WVAD()
        MAWVAD = self._MA(6, WVAD)
        return MAWVAD

    def REVS60(self):
        """
        因子描述：
        过去60天的价格动量。
        计算方法：

        close为收盘价。
        注1：若公司在过去的60天内有停牌，停牌日也计算在统计天数内;
        注2：若公司在今天停牌，不计算该因子的值；下同。
        """
        return self._REVS(60)

    def REVS120(self):
        """
        因子描述：
        过去120天的价格动量。
        计算方法：

        close为收盘价。
        注1：若公司在过去的120天内有停牌，停牌日也计算在统计天数内;
        注2：若公司在今天停牌，不计算该因子的值；下同。
        """
        return self._REVS(120)

    def REVS250(self):
        """
        因子描述：
        过去250天的价格动量。
        计算方法：

        close为收盘价。
        注1：若公司在过去的250天内有停牌，停牌日也计算在统计天数内;
        注2：若公司在今天停牌，不计算该因子的值；下同。
        """
        return self._REVS(250)

    def REVS750(self):
        """
        因子描述：
        过去750天的价格动量。
        计算方法：

        close为收盘价。
        注1：若公司在过去的750天内有停牌，停牌日也计算在统计天数内;
        注2：若公司在今天停牌，不计算该因子的值；下同。
        """
        return self._REVS(750)

    def REVS5m20(self):
        """
        因子描述：
        过去5天的价格动量减去过去1个月的价格动量。
        计算方法：

        close为收盘价
        """
        return self._REVS(5)-self._REVS(20)

    def REVS5m60(self):
        """
        因子描述：
        过去5天的价格动量减去过去3个月的价格动量。
        计算方法：

        close为收盘价
        """
        return self._REVS(5)-self._REVS(60)


    def REVS5Indu1(self):
        """
        因子描述：
        过去5日收益率与行业均值比较，行业均值等于所属行业所有个股5日收益率的等权平均，行业分类按照申万一级。
        计算方法：

        为该股票所处申万一级行业的行业均值
        close为收盘价
        注：由于个股当日停牌是取不到REVS5的值，所以直接从数据库中读取当日所有REVS5，然后按行业groupby取Mean就可以得到平均值；
        """

    def REVS20Indu1(self):
        """
        因子描述：
        过去1个月收益率与行业均值比较，行业均值等于所属行业所有个股20日收益率的等权平均，行业分类按照申万一级。
        计算方法：

        为该股票所处申万一级行业的行业均值
        close为收盘价
        注：由于个股当日停牌是取不到REVS20的值，所以直接从数据库中读取当日所有REVS20，然后按行业groupby取Mean就可以得到平均值；
        """

    def Volumn1M(self):
        """
        因子描述：
        当前交易量相比过去1个月日均交易量与过去一个月的收益率乘积。
        计算方法：

        其中 REVS20 指过去1个月的价格动量，Volumn 指今日成交量， 指过去20天的日成交量。
        注：上式为在个股在过去20天内不发生停牌的情况下，若个股有停牌，则成交量为0，那么在计算成交量的均值时需要去掉停牌日.
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        REVS20 = self._REVS(20)
        adj_volume = self.df_OHLC['volume']
        volumn1m = (20 * adj_volume / adj_volume.rolling(window=20, center=False).sum() - 1) * REVS20
        return volumn1m

    def Volumn3M(self):
        """
        因子描述：
        过去5天成交量相比过去三个月里五日成交量均值与过去三个月的收益率乘积。
        计算方法：
        其中 REVS60 指 过去3个月的价格动量，Volumn 指今日成交量， 指过去20天的日成交量。
        注：上式为在个股在过去20天内不发生停牌的情况下，若个股有停牌，则成交量为0，那么在计算成交量的均值时需要去掉停牌日
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        REVS60 = self._REVS(60)
        adj_volume = self.df_OHLC['volume']
        volumn3m = (12 * adj_volume.rolling(window=5, center=False).sum() / adj_volume.rolling(window=60,
                                                                                               center=False).sum() - 1) * REVS60
        return volumn3m

    def Price1M(self):
        """
        因子描述：
        当前股价除以过去1个月股价均值再减1，1个月按20个交易日计算
        计算方法：
        最近 日内的真实交易日期
        注：N=20， 为过去N天内，有交易的天数
        符号说明：
        符号 描述 计算方法
        close 今收盘价 Latest
        close_t-i 过去某天的收盘价 Latest
        """
        adj_close = self.df_OHLC['close']
        Price1M = 20 * adj_close / adj_close.rolling(window=20, center=False).sum() - 1
        return Price1M

    def Price3M(self):
        """
        因子描述：
        当前股价除以过去3个月股价均值再减1，1个月按20个交易日计算
        计算方法：
        最近 日内的真实交易日期
        注：N=60， 为过去N天内，有交易的天数
        符号说明：
        符号 描述 计算方法
        close 今收盘价 Latest
        close_t-i 过去某天的收盘价 Latest
        """
        adj_close = self.df_OHLC['close']
        Price3M = 60 * adj_close / adj_close.rolling(window=60, center=False).sum() - 1
        return Price3M

    def Price1Y(self):
        """
        因子描述：
        当前股价除以过去1年股价均值再减1，1年按250个交易日计算
        计算方法：
        最近 日内的真实交易日期
        注：N=250， 为过去N天内，有交易的天数
        符号说明：
        符号 描述 计算方法
        close 今收盘价 Latest
        close_t-i 过去某天的收盘价 Latest
        """
        adj_close = self.df_OHLC['close']
        Price1Y = 250 * adj_close / adj_close.rolling(window=250, center=False).sum() - 1
        return Price1Y

    def Rank1M(self):
        """
        因子描述：
        1减去过去一个月收益率排名与股票总数的比值。
        计算方法：
        注1：N为所有参与排名的股票数，剔除掉当日停牌的股票；
        注2：计算时，直接取出当日所有的REVS20，然后按照REVS20的值进行排名。
        符号说明：
        符号 描述
        Rank_value 过去1个月收益率的排名值（收益最高排名为1，最小为N）
        N 参与排名的所有A股数量
        """
        # REVS20 = self._REVS(20)
        # rank_value = REVS20.rank(axis=1, ascending=False)
        # N = len(REVS20.columns)
        # Rank1M = 1 - rank_value / N
        # return Rank1M

    # def UPDATE_TIME(self):
    #     pass
