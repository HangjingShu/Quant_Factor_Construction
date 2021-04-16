
import pandas as pd
import numpy as np
import statsmodels.api as sm
import multiprocessing

"""
get series from df_bic
calculate if necessary
drop duplicated records
expand the index list of df_bic by df_sb
div df_sb if necessary
"""


# 股票因子_趋势型 39
class EquFactorTrend(Equ):
    def __init__(self, OHLC, if_qfq=True):
        super().__init__()
        self.OHLC = OHLC
        self.if_qfq = if_qfq

    def _MA(self, N, any_series):
        MA = any_series.rolling(window=N, center=False, min_periods=1).mean()
        return MA

    def _EMA(self, N, date_length, adj_close):
        if date_length == 1:
            return adj_close
        else:
            alpha = 2 / (N + 1)
            EMA = alpha * adj_close + (1 - alpha) * self._EMA(N, date_length - 1, adj_close.shift(1))
            return EMA

    def AD(self):
        """
        因子描述：
        累积/派发线（Accumulation / Distribution Line）。该指标将每日的成交量通过价格加权累计，用以计算成交量的动
        通联数据内部资料，谨呈【木石投资】使用量。
        计算方法：
        1. Money Flow Volume = ((close - lowest) - (highest - close)) / (highest - lowest) * volume
        2. 若遇到最高价等于最低价（如直接封停板）：Money Flow Volume = (close / prev_close - 1) * volume。
        3. AD = 昨日AD + 今日Money Flow Volume，最后数值除以1e6。
        """
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        adj_close = self.df_OHLC['close']
        adj_volume = self.df_OHLC['volume']
        r = self.df_OHLC['pct_chg']
        MFV = ((adj_close - adj_low) - (adj_high - adj_close)) / (adj_high - adj_low) * adj_volume
        MFV = MFV.mask(adj_high == adj_low, r * adj_volume)
        MFV.iloc[:] = MFV.iloc[:].where(MFV.iloc[:].notnull(), 0)
        AD = MFV.cumsum() / 1e6
        return AD

    def AD20(self):
        """
        因子描述：
        20日累积/派发线（20-days Accumulation / Distribution Line）。该指标将每日的成交量通过价格加权累计，用以计
        算成交量的动量。
        计算方法：
        1. Money Flow Volume = ((close - lowest) - (highest - close)) / (highest - lowest) * volume
        2. 若遇到最高价等于最低价（如直接封停板）：Money Flow Volume = (close / prev_close - 1) * volume。
        3. AD = 昨日A + 今日Money Flow Volume
        4. AD20 = MA(AD,20)，最后数值除以1e6。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        adj_close = self.df_OHLC['close']
        adj_volume = self.df_OHLC['volume']
        r = self.df_OHLC['pct_chg']
        MFV = ((adj_close - adj_low) - (adj_high - adj_close)) / (adj_high - adj_low) * adj_volume
        MFV = MFV.mask(adj_high == adj_low, r * adj_volume)
        MFV.iloc[:] = MFV.iloc[:].where(MFV.iloc[:].notnull(), 0)
        temp = MFV.cumsum()
        AD = self._MA(20, temp) / 1e6
        return AD

    def AD6(self):
        """
        因子描述：
        6日累积/派发线（6-days Accumulation / Distribution Line）。该指标将每日的成交量通过价格加权累计，用以计算成
        交量的动量。
        计算方法：
        1. Money Flow Volume = ((close - lowest) - (highest - close)) / (highest - lowest) * volume
        2. 若遇到最高价等于最低价（如直接封停板）：Money Flow Volume = (close / prev_close - 1) * volume。
        3. AD = 昨日AD + 今日Money Flow Volume，最后数值除以1e6。
        4. AD6 = MA(AD,6)，最后数值除以1e6。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        adj_close = self.df_OHLC['close']
        adj_volume = self.df_OHLC['volume']
        r = self.df_OHLC['pct_chg']
        MFV = ((adj_close - adj_low) - (adj_high - adj_close)) / (adj_high - adj_low) * adj_volume
        MFV = MFV.mask(adj_high == adj_low, r * adj_volume)
        MFV.iloc[:] = MFV.iloc[:].where(MFV.iloc[:].notnull(), 0)
        temp = MFV.cumsum()
        AD = self._MA(6, temp) / 1e6
        return AD

    def _WMA_forCoppock(self, N, date_length, adj_close):
        if date_length == 1:
            return adj_close
        else:
            return N * adj_close + self._WMA_forCoppock(N, date_length - 1, adj_close.shift(1))

    def CoppockCurve(self):
        """
        因子描述：
        估波指标（Coppock Curve），又称“估波曲线”。该指标通过计算月度价格的变化速率的加权平均值来测量市场的动
        量，属于长线指标。这里我们改为日间的指标
        """
        adj_close = self.df_OHLC['close']
        RC = 100 * (adj_close / adj_close.shift(14) + adj_close / adj_close.shift(11))
        temp = (10 + 1) * 10 / 2
        Coppock = self._WMA_forCoppock(10, 10, RC) / temp
        return Coppock

    def ASI(self):
        """
        因子描述：
        累计振动升降指标（Accumulation Swing Index），又称实质线。ASI企图以开盘、最高、最低、收盘价构筑成一条幻
        想线，以便取代目前的走势，形成最能表现当前市况的真实市场线（Real Market）。
        计算方法：
        1. A = abs(highest – prev_close)
        2. B = abs(lowest – prev_close)
        3. C = abs(highest – prev_lowest)
        4. D = abs(prev_close – prev_open)
        5. E = close – prev_close
        6. F = close – open
        7. G = prev_close – prev_open
        8. X = E + F / 2 + G
        9. K = max(A, B)
        10. 比较A、B、C三者数值，若A最大，R=A + B / 2 + D / 4; 若B最大，R=A / 2 + B + D / 4；若C最大，R = C + D /
        4。
        11. SI = 16 * X / R * K
        12.
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        SI = self.SwingIndex()
        ASI = SI.rolling(window=6, center=False, min_periods=1).sum()
        return ASI

    def ChaikinOscillator(self):
        """
        因子描述：
        佳庆指标（Chaikin Oscillator）。该指标基于AD曲线的指数移动均线而计算得到。
        计算方法：
        ChaikinOscillator = EMA(AD, 3) – EMA(AD, 10)，最后数值除以1e6。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        AD = self.AD()
        ChaikinOscillator = self._EMA(3, 3, AD) - self._EMA(10, 10, AD)
        return ChaikinOscillator

    def ChaikinVolatility(self):
        """
        因子描述：
        佳庆离散指标（Chaikin Volatility，简称CVLT，VCI，CV），又称“佳庆变异率指数”，是通过测量一段时间内价格幅
        度平均值的变化来反映价格的离散程度。
        计算方法：
        1. HLEMA = EMA(highest - lowest, 10)
        2. ChaikinVolatility = 100 * (HLEMA (t) - HLEMA (t-10)) / HLEMA (t-10)
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        HLEMA = self._EMA(10, 10, adj_high - adj_low)
        ChaikinVolatility = 100 * HLEMA.diff(10) / HLEMA.shift(10)
        return ChaikinVolatility

    def EMV14(self):
        """
        因子描述：
        简易波动指标（14-days Ease of Movement Value）。EMV将价格与成交量的变化结合成一个波动指标来反映股价或
        指数的变动状况。由于股价的变化和成交量的变化都可以引发该指标数值的变动，EMV实际上也是一个量价合成指标。
        计算方法：
        EMV =MA(((highest + lowest) / 2 – (prev_highest + prev_lowest) / 2) * (highest – lowest) / volume, N)
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        adj_volume = self.df_OHLC['volume']
        temp = ((adj_high + adj_low) / 2 - (adj_high.shift(1) + adj_low.shift(1)) / 2) * \
               (adj_high - adj_low) / adj_volume
        EMV = self._EMA(14, 14, temp)
        return EMV

    def EMV6(self):
        """
        因子描述：
        简易波动指标（6-days Ease of Movement Value）。EMV将价格与成交量的变化结合成一个波动指标来反映股价或指
        数的变动状况。由于股价的变化和成交量的变化都可以引发该指标数值的变动，EMV实际上也是一个量价合成指标。
        计算方法：
        EMV =MA(((highest + lowest) / 2 – (prev_highest + prev_lowest) / 2) * (highest – lowest) / volume, N)
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        adj_volume = self.df_OHLC['volume']
        temp = ((adj_high + adj_low) / 2 - (adj_high.shift(1) + adj_low.shift(1)) / 2) * \
               (adj_high - adj_low) / adj_volume
        EMV = self._EMA(6, 6, temp)
        return EMV

    def _PlusDM(self, adj_high):
        """
        参照网上资料定义
        """
        plusdm = adj_high - adj_high.shift(1)
        plusdm = plusdm.mask(plusdm < 0, 0)
        return plusdm

    def plusDI(self):
        """
        因子描述：
        上升指标 (Plus directional indicator)，DMI因子的构成部分。
        计算方法：
        1. TR = max(highest-lowest, abs(highest-prev_close), abs(lowest-prev_close))
        2. UpMove = highest - pre_highest , DownMove = pre_lowest - lowest
        3. 当 UpMove > DownMove 且 UpMove > 0时, plusDM = UpMove, 否则plusDM = 0
        4. , N=14
        """
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        adj_close = self.df_OHLC['close']
        TH = adj_high.mask(adj_close.shift(1) > adj_high, adj_close.shift(1))
        TL = adj_low.mask(adj_close.shift(1) < adj_low, adj_close.shift(1))
        TR = TH - TL
        plusdi = self._EMA(14, 14, self._PlusDM(adj_high)) / self._EMA(14, 14, TR)
        return plusdi

    def _MinusDM(self, adj_low):
        """
        参照网上资料定义
        """
        minusdm = adj_low.shift(1) - adj_low
        minusdm = minusdm.mask(minusdm < 0, 0)
        return minusdm

    def minusDI(self):
        """
        因子描述：
        下降指标 (Minus directional indicator)，DMI因子的构成部分。
        计算方法：
        1. TR = max(highest-lowest, abs(highest-prev_close), abs(lowest-prev_close))
        2. UpMove = highest - pre_highest , DownMove = pre_lowest - lowest
        3. 当 UpMove < DownMove 且 DownMove > 0时, minusDM = DownMove, 否则minusDM = 0
        4. , N = 14
        """
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        adj_close = self.df_OHLC['close']
        TH = adj_high.mask(adj_close.shift(1) > adj_high, adj_close.shift(1))
        TL = adj_low.mask(adj_close.shift(1) < adj_low, adj_close.shift(1))
        TR = TH - TL
        minusdi = self._EMA(14, 14, self._MinusDM(adj_low)) / self._EMA(14, 14, TR)
        return minusdi

    def ADX(self):
        """
        因子描述：
        平均动向指数 (Average directional index)，DMI因子的构成部分。
        计算方法：
        N = 14
        """
        plusdi = self.plusDI()
        minusdi = self.minusDI()
        DX = (plusdi - minusdi).abs() / (plusdi + minusdi)
        ADX = self._EMA(14, 14, DX)
        return ADX

    def ADXR(self):
        """
        因子描述：
        相对平均动向指数 (Relative average directional index)，DMI因子的构成部分。
        计算方法：
        N = 14
        """
        adx1 = self.ADX()
        adx2 = adx1.shift(14)
        ADXR = (adx1 + adx2) / 2
        return ADXR

    def Aroon(self):
        """
        因子描述：
        Aroon (Aroon oscillator) 通过计算自价格达到近期最高值和最低值以来所经过的期间数，帮助投资者预测证券价格从趋
        势到区域区域或反转的变化。Aroon指标分为Aroon、AroonUp和AroonDown3个具体指标。
        计算方法：
        若考察一段长度为N（一般取26个交易日）的区间，以x表示该区间最高价出现日距离当前交易日的天数，以y表示该区间
        最低价出现日距离当前交易日的天数。
        """
        aroonup = self.AroonUp()
        aroondown = self.AroonDown()
        aroon = aroonup - aroondown
        return aroon

    def AroonDown(self):
        """
        因子描述：
        计算Aroon因子的中间变量 (Mediator in calculating Aroon, )。
        计算方法：
        若考察一段长度为N（取前26个非停牌的交易日）的区间，以y表示该区间最低价出现日距离当前交易日的交易天数（即
        停牌日和非开盘日不计算在内）。
        """
        for_aroon_down = lambda x: pd.Series(x).idxmin(axis=1)
        date_length = 26
        adj_low = self.df_OHLC['low'].copy(deep=True)
        adj_low.index = range(len(adj_low.index))
        y = date_length - adj_low.rolling(window=date_length, center=False).apply(func=for_aroon_down) - 1
        AroonDown = (date_length - y) / date_length
        AroonDown.index = self.df_OHLC['low'].index
        return AroonDown

    def AroonUp(self):
        """
        因子描述：
        计算Aroon因子的中间变量 (Mediator in calculating Aroon) 。
        计算方法：
        若考察一段长度为N（取前26个非停牌的交易日）的区间，以x表示该区间最高价出现日距离当前交易日的交易天数（即
        停牌日和非开盘日不计算在内）。
        """
        for_aroon_up = lambda x: pd.Series(x).idxmax(axis=1)
        date_length = 26
        adj_high = self.df_OHLC['high'].copy(deep=True)
        adj_high.index = range(len(adj_high.index))
        x = date_length - adj_high.rolling(window=date_length, center=False).apply(func=for_aroon_up) - 1
        AroonUp = (date_length - x) / date_length
        AroonUp.index = self.df_OHLC['low'].index
        return AroonUp

    def DEA(self):
        """
        因子描述：
        计算MACD因子的中间变量 (Difference in Exponential Average（mediator in calculating MACD))。
        计算方法：
        1. DIFF（Difference）为收盘价短期、长期指数平滑移动平均线间的差：DIFF = EMA(close, 12) – EMA(close, 26)。
        2. DEA为（Difference Exponential Average）DIFF的M日指数移动平均：DEA = EMA(DIFF, M)，通常M = 9。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        DEA = self._EMA(9, 9, self.DIFF())
        return DEA

    def DIFF(self):
        """
        因子描述：
        计算MACD因子的中间变量 (Difference（mediator in calculating MACD)）。
        计算方法：
        DIFF（Difference）为收盘价短期、长期指数平滑移动平均线间的差。
        DIFF = EMA(close, 12) – EMA(close, 26)。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        DIFF = self._EMA(12, 12, adj_close) - self._EMA(26, 26, adj_close)
        return DIFF

    def _DMZ(self):
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        temp1 = adj_high + adj_low
        temp2 = adj_high.shift(1) + adj_low.shift(1)
        temp3 = (adj_high - adj_high.shift(1)).abs()
        temp4 = (adj_low - adj_low.shift(1)).abs()
        DMZ = temp3.mask(temp4 > temp3, temp4)
        DMZ = DMZ.mask(temp1 <= temp2, 0)
        return DMZ

    def _DMF(self):
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        temp1 = adj_high + adj_low
        temp2 = adj_high.shift(1) + adj_low.shift(1)
        temp3 = (adj_high - adj_high.shift(1)).abs()
        temp4 = (adj_low - adj_low.shift(1)).abs()
        DMF = temp3.mask(temp4 > temp3, temp4)
        DMF = DMF.mask(temp1 >= temp2, 0)
        return DMF

    def DDI(self):
        """
        因子描述：
        方向标准离差指数 (Directional Divergence Index)。观察一段时间内股价相对于前一天向上波动和向下波动的比例，并
        对其进行移动平均分析。DDI指标倾向于显示一种长波段趋势的方向改变。
        计算方法：
        1. 若(highest + lowest) <= (prev_highest + prev_lowest)，DMZ = 0
        2. 若(highest + lowest) > (prev_highest + prev_lowest)，DMZ = max(abs(highest – prev_highest), abs(lowest –
        prev_lowest))
        3. 若(highest + lowest) >= (prev_highest + prev_lowest)，DMF = 0
        4. 若(highest + lowest) < (prev_highest + prev_lowest)，DMF = max(abs(highest – prev_highest), abs(lowest –
        prev_lowest))
        5. DIZ = SUM(DMZ, N) / (SUM(DMZ, N) + SUM(DMF, N))
        6. DIF = SUM(DMF, N) / (SUM(DMZ, N) + SUM(DMF, N))
        7. DDI = DIZ – DIF
        取N = 13。
        """
        DIZ = self.DIZ()
        DIF = self.DIF()
        DDI = DIZ - DIF
        return DDI

    def DIZ(self):
        """
        因子描述：
        计算DDI因子的中间变量 (Mediator in calculating DDI)。
        计算方法：
        1. 若(highest + lowest) <= (prev_highest + prev_lowest)，DMZ = 0
        2. 若(highest + lowest) > (prev_highest + prev_lowest)，DMZ = max(abs(highest – prev_highest), abs(lowest –
        prev_lowest))
        3. 若(highest + lowest) >= (prev_highest + prev_lowest)，DMF = 0
        4. 若(highest + lowest) < (prev_highest + prev_lowest)，DMF = max(abs(highest – prev_highest), abs(lowest –
        prev_lowest))
        5. DIZ = SUM(DMZ, N) / (SUM(DMZ, N) + SUM(DMF, N))，取N=13.
        """
        date_length = 13
        DMZ = self._DMZ()
        DMF = self._DMF()
        DIZ = DMZ.rolling(window=date_length, center=False).sum() / (
                DMZ.rolling(window=date_length, center=False).sum() +
                DMF.rolling(window=date_length, center=False).sum())
        return DIZ

    def DIF(self):
        """
        因子描述：
        计算DDI因子的中间变量 (Mediator in calculating DDI)。
        计算方法：
        1. 若(highest + lowest) <= (prev_highest + prev_lowest)，DMZ = 0
        2. 若(highest + lowest) > (prev_highest + prev_lowest)，DMZ = max(abs(highest – prev_highest), abs(lowest –
        prev_lowest))
        3. 若(highest + lowest) >= (prev_highest + prev_lowest)，DMF = 0
        4. 若(highest + lowest) < (prev_highest + prev_lowest)，DMF = max(abs(highest – prev_highest), abs(lowest –
        prev_lowest))
        5. DIF = SUM(DMF, N) / (SUM(DMZ, N) + SUM(DMF, N))，取N=13.
        """
        date_length = 13
        DMZ = self._DMZ()
        DMF = self._DMF()
        DIF = DMF.rolling(window=date_length, center=False).sum() / (
                DMZ.rolling(window=date_length, center=False).sum() +
                DMF.rolling(window=date_length, center=False).sum())
        return DIF

    def MACD(self):
        """
        因子描述：
        平滑异同移动平均线（Moving Average Convergence Divergence）,又称移动平均聚散指标。
        计算方法：
        1. DIFF（Difference）为收盘价短期、长期指数平滑移动平均线间的差：DIFF = EMA(close, 12) – EMA(close, 26)。
        2. DEA为（Difference Exponential Average）DIFF的M日指数移动平均：DEA = EMA(DIFF, M)，通常M = 9。
        3. MACD为DIFF和DEA之差，按照国内的处理标准，最终结果乘以2。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        MACD = 2 * (self.DIFF() - self.DEA())
        return MACD

    def MTM(self):
        """
        因子描述：
        动量指标（Momentom Index）。动量指数以分析股价波动的速度为目的，研究股价在波动过程中各种加速，减速，惯
        性作用以及股价由静到动或由动转静的现象。
        计算方法：
        , N = 10
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        MTM = adj_close - adj_close.shift(10)
        return MTM

    def MTMMA(self):
        """
        因子描述：
        因子mtm的10日均值 (10-day average momentum index)。
        计算方法：
        , N = 10
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        MTMMA = self._MA(10, self.MTM())
        return MTMMA

    def PVT(self):
        """
        因子描述：
        价量趋势（Price and Volume Trend）指标。把能量变化与价格趋势有机地联系到了一起，从而构成了量价趋势指标。
        计算方法：
        PVT_today = (close – prev_close) / prev_close * volume
        今日PVT = 昨日PVT + PVT_today
        最后入库数值除以1e6。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        date_length = 1
        adj_close = self.df_OHLC['close']
        adj_volume = self.df_OHLC['volume']
        PVT = (adj_close - adj_close.shift(1)) / adj_close.shift(1) * adj_volume
        PVT_ac = PVT.rolling(window=date_length, center=False).sum() / 1e6
        return PVT_ac

    def PVT6(self):
        """

        """
        date_length = 6
        adj_close = self.df_OHLC['close']
        adj_volume = self.df_OHLC['volume']
        PVT = (adj_close - adj_close.shift(1)) / adj_close.shift(1) * adj_volume
        PVT_ac = PVT.rolling(window=date_length, center=False).sum() / 1e6
        return PVT_ac

    def PVT12(self):
        """
        """
        date_length = 12
        adj_close = self.df_OHLC['close']
        adj_volume = self.df_OHLC['volume']
        PVT = (adj_close - adj_close.shift(1)) / adj_close.shift(1) * adj_volume
        PVT_ac = PVT.rolling(window=date_length, center=False).sum() / 1e6
        return PVT_ac

    def _TRIX(self, date_length, adj_close):
        """
        因子描述：
        5日三重指数平滑移动平均指标变化率（5-day percent rate of change of triple exponetially smoothed moving
        average）。TRIX根据移动平均线理论，对一条平均线进行三次平滑处理，再根据这条移动平均线的变动情况来预测股价
        的长期走势。
        计算方法：
        1. EMA3 = EMA(EMA(EMA(close, N), N), N)
        2. TRIX = EMA3(t) / EMA3(t-1) – 1
        N取5
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        EMA3 = self._EMA(date_length, date_length, self._EMA(date_length, date_length,
                                                             self._EMA(date_length, date_length, adj_close)))
        TRIX = EMA3 / EMA3.shift(1) - 1
        return TRIX

    def TRIX5(self):
        """
        因子描述：
        5日三重指数平滑移动平均指标变化率（5-day percent rate of change of triple exponetially smoothed moving
        average）。TRIX根据移动平均线理论，对一条平均线进行三次平滑处理，再根据这条移动平均线的变动情况来预测股价
        的长期走势。
        计算方法：
        1. EMA3 = EMA(EMA(EMA(close, N), N), N)
        2. TRIX = EMA3(t) / EMA3(t-1) – 1
        N取5
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        return self._TRIX(5, adj_close)

    def TRIX10(self):
        """
        因子描述：
        10日三重指数平滑移动平均指标变化率（10-day percent rate of change of triple exponetially smoothed moving
        average）。TRIX根据移动平均线理论，对一条平均线进行三次平滑处理，再根据这条移动平均线的变动情况来预测股价
        的长期走势。
        计算方法：
        1. EMA(EMA(EMA(close, N), N), N)
        2. TRIX = EMA3(t) / EMA3(t-1) – 1
        N取10。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        return self._TRIX(10, adj_close)

    def UOS(self):
        """
        因子描述：
        终极指标（Ultimate Oscillator）。现行使用的各种振荡指标，对于周期参数的选择相当敏感。不同市况、不同参数设定
        的振荡指标，产生的结果截然不同。因此，选择最佳的参数组合，成为使用振荡指标之前最重要的一道手续。
        计算方法：
        1. TH = max(highest, prev_close)
        2. TL = min(lowest, prev_close)
        3. TR = TH – TL
        4. XR = close – TL
        5. XRM = M日XR之和 / M日TR之和
        6. XRN = N日XR之和 / N日TR之和
        7. XRO = O日XR之和 / O日TR之和
        8. UOS = 100 * (XRM*N*O + XRN*M*O + XRO*M*N) / (M*N + M*O+ N*O)
        一般M = 7, N = 14, O = 28
        """
        date_length_M = 7
        date_length_N = 14
        date_length_O = 28
        adj_close = self.df_OHLC['close']
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        TH = adj_high.mask(adj_close.shift(1) > adj_high, adj_close.shift(1))
        TL = adj_low.mask(adj_close.shift(1) < adj_low, adj_close.shift(1))
        TR = TH - TL
        XR = adj_close - TL
        XRM = XR.rolling(window=date_length_M, center=False).sum() / TR.rolling(window=date_length_M,
                                                                                center=False).sum()
        XRN = XR.rolling(window=date_length_N, center=False).sum() / TR.rolling(window=date_length_N,
                                                                                center=False).sum()
        XRO = XR.rolling(window=date_length_O, center=False).sum() / TR.rolling(window=date_length_O,
                                                                                center=False).sum()
        UOS = 100 * (
                XRM * date_length_N * date_length_O + XRN * date_length_M * date_length_O +
                XRO * date_length_M * date_length_N) / (
                      date_length_N * date_length_M + date_length_N * date_length_O + date_length_M * date_length_O)
        return UOS

    def _MA10RegressCoeff(self, date_length, adj_close, ascending=True):
        """
          因子描述：
          N日价格平均线N日线性回归系数 (regression coefficient of 10-day moving average （in predicting 12-day
          moving average）)。
          计算方法：
          取近N个交易日的对应的MA值，对N个周期的序数进行的普通最小二乘的线性回归，取股价关于周期序数的系数。
          式中 即为MA10RegressCoeff。N取12日。
          注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
          情数据计算得到
        """
        if ascending:
            a = list(range(1, date_length + 1))
        else:
            a = list(range(date_length, 0, -1))
        MA = self._MA(date_length, adj_close)
        fit = lambda t: sm.OLS(t, sm.add_constant(a)).fit().params[1]
        MARegressCoef = MA.rolling(window=date_length, center=False).apply(func=fit)
        return MARegressCoef

    def MA10RegressCoeff12(self):
        """
        因子描述：
        10日价格平均线N日线性回归系数 (regression coefficient of 10-day moving average （in predicting 12-day
        moving average）)。
        计算方法：
        取近N个交易日的对应的MA值，对N个周期的序数进行的普通最小二乘的线性回归，取股价关于周期序数的系数。
        式中 即为MA10RegressCoeff。N取12日。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        return self._MA10RegressCoeff(12, adj_close)

    def MA10RegressCoeff6(self):
        """
        因子描述：
        10日价格平均线N日线性回归系数 (regression coefficient of 10-day moving average （in predicting 12-day
        moving average）)。
        计算方法：
        取近N个交易日的对应的MA值，对N个周期的序数进行的普通最小二乘的线性回归，取股价关于周期序数的系数。
        式中 即为MA10RegressCoeff。N取6日。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        return self._MA10RegressCoeff(6, adj_close)

    def _PLRC(self, date_length, adj_close, ascending=True):
        """
        因子描述：
        价格线性回归系数（6-day Price Linear Regression Coefficient）。
        计算方法：
        取近N个交易日的收盘价对交易日的序数做普通最小二乘的线性回归。取股价关于周期序数的系数作为因子。
        式中 即为PLRC。N取6。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        if ascending:
            a = list(range(1, date_length + 1))
        else:
            a = list(range(date_length, 0, -1))
        fit = lambda t: sm.OLS(t, sm.add_constant(a)).fit().params[1]
        PLRC = adj_close.rolling(window=date_length, center=False).apply(func=fit)
        return PLRC

    def PLRC6(self):
        """
        因子描述：
        价格线性回归系数（6-day Price Linear Regression Coefficient）。
        计算方法：
        取近N个交易日的收盘价对交易日的序数做普通最小二乘的线性回归。取股价关于周期序数的系数作为因子。
        式中 即为PLRC。N取6。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        return self._PLRC(6, adj_close)

    def PLRC12(self):
        """
        因子描述：
        价格线性回归系数（12-day Price Linear Regression Coefficient）。
        计算方法：
        取近N个交易日的收盘价对交易日的序数做普通最小二乘的线性回归。取股价关于周期序数的系数作为因子。
        式中 即为PLRC。N取12。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        return self._PLRC(12, adj_close)

    def SwingIndex(self):
        """
        因子描述：
        振动升降指标。计算ASI因子的中间变量 (Swing Index)。
        计算方法：
        1. A = |hihgest – prev_close|
        B = |lowest – prev_close|
        C = |highest – prev_lowest|
        D = |prev_close – prev_open|
        2. E = close – prev_close
        F = close – open
        G = prev_close – prev_open
        3. X = E + F / 2 + G
        4. K = max(A, B)
        5. 比较A、B、C三者数值，若A最大，R=A + B / 2 + D / 4; 若B最大，R=A / 2 + B + D / 4；若C最大，R = C + D /
        4。
        6. SI = 16 * X / R * K
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行
        情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        adj_open = self.df_OHLC['open']
        A = (adj_close - adj_close.shift(1)).abs()
        B = (adj_low - adj_close.shift(1)).abs()
        C = (adj_high - adj_low.shift(1)).abs()
        D = (adj_close.shift(1) - adj_open.shift(1)).abs()
        E = adj_close - adj_close.shift(1)
        F = adj_close - adj_open
        G = adj_close.shift(1) - adj_open.shift(1)
        X = E + F / 2 + G
        K = B.mask(A > B, A)
        R1 = A + B / 2 + D / 4
        R2 = A / 2 + B + D / 4
        R3 = C + D / 4
        MAX = A.mask(B > A, B)
        MAX = MAX.mask(C > MAX, C)
        R = R1
        R = R.mask(MAX == B, R2)
        R = R.mask(MAX == C, R3)
        SI = 16 * X / R * K
        return SI

    def Ulcer5(self):
        """
        因子描述：
        (5-day Ul index), 由Peter Martin于1987年提出，1989年发表于Peter Martin和Byron McCann的著作The Investors
        Guide to Fidelity Funds。用于考察向下的波动性。
        计算方法：
        N取5
        """
        date_length = 5
        adj_close = self.df_OHLC['close']
        R = (adj_close - adj_close.rolling(window=date_length, center=False).max()) / adj_close.rolling(
            window=date_length, center=False).max()
        Ulcer = np.square(R)
        Ulcer = Ulcer.rolling(window=date_length, center=False).sum() / date_length
        Ulcer = Ulcer.apply(np.sqrt)
        return Ulcer

    def Ulcer10(self):
        """
        因子描述：
        (5-day Ul index), 由Peter Martin于1987年提出，1989年发表于Peter Martin和Byron McCann的著作The Investors
        Guide to Fidelity Funds。用于考察向下的波动性。
        计算方法：
        N取10
        """
        date_length = 10
        adj_close = self.df_OHLC['close']
        R = (adj_close - adj_close.rolling(window=date_length, center=False).max()) / adj_close.rolling(
            window=date_length, center=False).max()
        Ulcer = np.square(R)
        Ulcer = Ulcer.rolling(window=date_length, center=False).sum() / date_length
        Ulcer = np.sqrt(Ulcer)
        return Ulcer

    def DHILO(self):
        """
        因子描述：
        波幅中位数（median of volatility），每日对数最高价和对数最低价差值的3 月内中位数。
        计算方法：
        最高价和最低价缺失时用收盘价填补。
        符号说明：
        符号 描述 计算方法
        highest 最高价 Latest
        lowest 最低价 Latest
        close 今收价 Latest
        三个月处理为60个交易日
        """
        adj_low = self.df_OHLC['low']
        adj_high = self.df_OHLC['high']
        temp1 = np.log(adj_high)
        temp2 = np.log(adj_low)
        DHILO = (temp1 - temp2).rolling(window=60, center=False).median()
        return DHILO

    # @timer
    # def _Hurst(self, N):
    #     """
    #     因子描述：
    #     赫斯特指数（Hurst exponent）。是由英国水文专家H．E．Hurst提出了用重标极差(R/S)分析方法来建立赫斯特指数
    #     (H)，作为判断时间序列数据遵从随机游走还是有偏的随机游走过程的指标。当H > 0.5时，时间序列呈现出持续性；当H
    #     = 0.5时，序列呈现出随机游走；当H < 0.5时，表现出反持续性。
    #     计算方法：
    #     第一步：将价格时间序列划分成多个实际长度为n的子区间集，分别取n属于[5,10,20,40,80];
    #     第二步：对于一个固定的n，遍历其中每一个日期子集j,计算 ，最后求均值
    #     计算详情：
    #     """
    #     Hurst = self.df_OHLC['close'].copy()
    #
    #     def get_Hurst(x):
    #         adj_close_tmp = x
    #         X = np.log(adj_close_tmp) - np.log(adj_close_tmp.shift(1))
    #         X = X.drop(index=X.index[0])
    #         array = [len(X), len(X) / 2, len(X) / 4, len(X) / 8, len(X) / 16]
    #         Y = []
    #         for n in array:
    #             y = []
    #             n = int(n)
    #             for i in range(int(len(X) / n)):
    #                 i = int(i)
    #                 mu = X[i * n:(i + 1) * n].mean()
    #                 sigma = X[i * n:(i + 1) * n].std()
    #                 Z = (X[i * n:(i + 1) * n] - mu)
    #                 Z_t = Z.cumsum()
    #                 R = Z_t.max() - Z_t.min()
    #                 try:
    #                     T = R / sigma
    #                 except ZeroDivisionError:
    #                     T = np.nan
    #                 y.append(T)
    #             y = pd.Series(y)
    #             Y.append(y.mean())
    #         mod = sm.OLS(Y, sm.add_constant(array))
    #         res = mod.fit()
    #         return res.params[1]
    #
    #     if len(Hurst) < N:
    #         raise Exception()
    #     else:
    #         Hurst = Hurst.rolling(window=(N + 1), min_periods=20).apply(get_Hurst)
    #         return Hurst
    #
    # @timer
    # def Hurst32(self):
    #     return self._Hurst(32)
    #
    # @timer
    # def Hurst64(self):
    #     return self._Hurst(64)
    #
    # @timer
    # def Hurst128(self):
    #     return self._Hurst(128)
    #
    # @timer
    # def Hurst256(self):
    #     return self._Hurst(256)

    # def UPDATE_TIME(self):
    #     pass
    # def run(self,table_name,code,dtype):
    #     try:
    #         self.df_OHLC = self.OHLC.fetch(stock_code=code, if_qfq=self.if_qfq)
    #     except Exception:
    #         logging.warning('{} fail'.format(code))
    #     df_final = self.loop_function_list(code)
    #     self.dbh.save_data_to_db(df_final, table_name, "append", dtype)

