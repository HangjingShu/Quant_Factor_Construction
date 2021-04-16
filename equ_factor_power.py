import numpy as np
import pandas as pd
import statsmodels.api as sm
from pyfinance import ols

"""
get series from df_bic
calculate if necessary
drop duplicated records
expand the index list of df_bic by df_sb
div df_sb if necessary
"""


# 股票因子_能量型 18
class EquFactorPower(Equ):
    def __init__(self, OHLC, Index_OHLC, sb, if_qfq=False):
        super().__init__()
        self.OHLC = OHLC
        self.sb = sb
        self.Index_OHLC = Index_OHLC
        self.if_qfq = if_qfq

    def AR(self):
        """
        因子描述：
        人气指标 (price movement indicator, compare buying power with selling power to open price)。是以当天开市价
        为基础，即以当天市价分别比较当天最高，最低价，通过一定时期内开市价在股价中的地位，反映市场买卖人气。
        计算方法：

        N = 26。
        """
        date_length = 26
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        adj_open = self.df_OHLC['open']
        AR = (adj_high - adj_open).rolling(window=date_length, center=False).sum() / \
             (adj_open - adj_low).rolling(window=date_length, center=False).sum()
        AR = AR * 100
        return AR

    def BR(self):
        """
        因子描述：
        意愿指标 (price movement indicator, compare buying power with selling power to last day close price)。是以昨日收市价为基础，分别与当
        日最高，最低价相比，通过一定时期收市收在股价中的地位，反映市场买卖意愿的程度。
        计算方法：
        N = 26。
        """
        date_length = 26
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        adj_close = self.df_OHLC['close']
        temp1 = adj_high - adj_close.shift(1)
        temp1 = temp1.mask(temp1 < 0, 0)
        temp2 = adj_close.shift(1) - adj_low
        temp2 = temp2.mask(temp2 < 0, 0)
        BR = temp1.rolling(window=date_length, center=False).sum() / \
             temp2.rolling(window=date_length, center=False).sum()
        BR = BR * 100
        return BR

    def ARBR(self):
        """
        因子描述：
        人气指标（AR）和意愿指标（BR）都是以分析历史股价为手段的技术指标 (Difference between AR and BR)。人气指标是以当天开市价为基础，即以当天市价分别比较当天最高，最低价，通过一定时期内开市价在股价中的地位，反映市场
        买卖人气；意愿指标是以昨日收市价为基础，分别与当日最高，最低价相比，通过一定时期收市收在股价中的地位，反映市场买卖意愿的程度。
        计算方法：

        N = 26。
        """
        ARBR = self.AR() - self.BR()
        return ARBR

    def CR20(self):
        """
        因子描述：
        CR指标以上一个计算周期（如N日）的中间价比较当前周期（如日）的最高价 (price movement indicator, compare buying power and selling power to previous mid
        price (20 days) ) 、最低价，计算出一段时期内股价的“强弱”。
        计算方法：
        1. TYP = (highest + lowest + close) / 3。
        2. CR = sum(max(highest – prev_typical, 0), N) / sum(max(prev_typical – lowest, 0), N) * 100
        取N = 20。
        """
        date_length = 20
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        adj_close = self.df_OHLC['close']
        TYP = (adj_low + adj_high + adj_close) / 3
        temp_long = adj_high - TYP.shift(1)
        temp_long1 = temp_long.mask(temp_long < 0, 0)
        temp_short = TYP.shift(1) - adj_low
        temp_short1 = temp_short.mask(temp_short < 0, 0)
        CR = 100 * temp_long1.rolling(window=date_length, center=False).sum() / temp_short1.rolling(window=20,
                                                                                                    center=False).sum()
        return CR

    def _EMA(self, N, date_length, adj_close):
        if date_length == 1:
            return adj_close
        else:
            alpha = 2 / (N + 1)
            EMA = alpha * adj_close + (1 - alpha) * self._EMA(N, date_length - 1, adj_close.shift(1))
            return EMA

    def MassIndex(self):
        """
        因子描述：
        梅斯线（Mass Index）。本指标是Donald Dorsey累积股价波幅宽度之后所设计的震荡曲线。其最主要的作用，在于寻找飙涨股或者极度弱势股的重要趋势反转点。
        计算方法：
        1. EMAHL = EMA(highest – lowest, 9)。
        2. EMA Ratio = EMAHL / EMA(EMAHL, 9)。
        3. MassIndex = EMA Ratio的25天的累加值。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        adj_high = self.df_OHLC['high']
        adj_low = self.df_OHLC['low']
        EMAHL = self._EMA(9, 9, adj_high - adj_low)
        EMA_Ratio = EMAHL / self._EMA(9, 9, EMAHL)
        MassIndex = EMA_Ratio.rolling(window=25, center=False).sum()
        return MassIndex

    def BearPower(self):
        """
        因子描述：
        空头力道(Mediator in calculating Elder, Bear power indicator)，是计算Elder因子的中间变量。
        计算方法：
        BearPower = lowest – EMA(close, N),其中N取13。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        adj_low = self.df_OHLC['low']
        BearPower = adj_low - self._EMA(13, 13, adj_close)
        return BearPower

    def BullPower(self):
        """
        因子描述：
        多头力道 (Mediator in calculating Elder, Bull power indicator)，是计算Elder因子的中间变量。
        计算方法：
        BullPower = highest – EMA(close, N),其中N取13。
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        adj_high = self.df_OHLC['high']
        BullPower = adj_high - self._EMA(13, 13, adj_close)
        return BullPower

    def Elder(self):
        """
        因子描述：
        艾达透视指标（Elder-ray Index）。交易者可以经由该指标，观察市场表面之下的多头与空头力道。
        计算方法：
        1. BullPower = highest – EMA(close, N)。
        2. BearPower = lowest – EMA(close, N)。
        3. Elder = (BullPower - BearPower) / close
        N取13。
        """
        adj_close = self.df_OHLC['close']
        BullPower = self.BullPower()
        BearPower = self.BearPower()
        Elder = (BullPower - BearPower) / adj_close
        return Elder

    def NVI(self):
        """
        因子描述：
        负成交量指标（Negative Volume Index）。本指标的主要作用是辨别目前市场行情是处于多头行情还是空头行情，并追踪市场资金流向。
        计算方法：
        1. 上市日第一天的NVI为100。
        2. 若当日的成交量小于前一日的成交量，则
        3. 若当日的成交量大于前一日的成交量，则
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        adj_volume = self.df_OHLC['volume']
        temp1 = adj_close / adj_close.shift(1)
        temp2 = adj_volume.diff(1)
        delta = temp1.mask(temp2 >= 0, np.nan)
        delta = delta.where(delta.notnull(), 1)
        NVI = delta.cumprod() * 100
        return NVI

    #  计算问题

    def PVI(self):
        """
        因子描述：
        正成交量指标（Positive Volume Index）。本指标的主要作用是辨别目前市场行情是处于多头行情还是空头行情，并追踪市场资金流向。
        计算方法：
        1. 上市日第一天的PVI为100。
        2. 若当日的成交量大于前一日的成交量，则
        3. 若当日的成交量小于前一日的成交量，则
        注：因子数据由动态前复权的行情数据计算得到，即用因子值当日的PIT行情数据，以及该日行情为基准的前复权历史行情数据计算得到
        """
        adj_close = self.df_OHLC['close']
        adj_volume = self.df_OHLC['volume']
        temp1 = adj_close / adj_close.shift(1)
        temp2 = adj_volume.diff(1)
        delta = temp1.mask(temp2 <= 0, np.nan)
        delta = delta.where(delta.notnull(), 1)
        PVI = delta.cumprod() * 100
        return PVI

    def _RC(self, N):
        """
        因子描述：
        N日变化率指数（12-day Rate of Change），类似于动力指数。如果价格始终是上升的，则变化率指数始终在100%线以上，且如果变化速度指数在向上发展时，说明价格上升的速度在加快。
        计算方法：

        N取12，close是收盘价
        注：若公司在过去的N天内有停牌，停牌日也计算在统计天数内;
        """
        adj_close = self.df_OHLC['close']
        RC = adj_close / adj_close.shift(N)
        return RC

    def RC12(self):
        """
        因子描述：
        12日变化率指数（12-day Rate of Change），类似于动力指数。如果价格始终是上升的，则变化率指数始终在100%线以上，且如果变化速度指数在向上发展时，说明价格上升的速度在加快。
        计算方法：

        N取12，close是收盘价
        注：若公司在过去的N天内有停牌，停牌日也计算在统计天数内;
        """
        return self._RC(12)

    def RC24(self):
        """
        因子描述：
        24日变化率指数（24-day Rate of Change），类似于动力指数。如果价格始终是上升的，则变化率指数始终在100%线以上，且如果变化速度指数在向上发展时，说明价格上升的速度在加快。
        计算方法：

        N取24，close是收盘价
        注：若公司在过去的N天内有停牌，停牌日也计算在统计天数内;
        """
        return self._RC(24)

    def _RSTR(self, N):
        """
        因子描述：
        N月相对强势（Relative strength for the last 24 months）。
        计算方法：

        其中N取24个自然月对应的时间长度，r表示日收益。计算中将每日无风险收益 按0 处理。
        """
        temp = self.df_OHLC['pct_chg']
        temp1 = (temp / 100 + 1).apply(np.log)
        temp2 = pd.Series(np.log(1), index=self.df_OHLC.index)
        RSTRN = temp1.rolling(window=N, center=False).sum() - temp2.rolling(window=N, center=False).sum()
        return RSTRN

    def RSTR24(self):
        """
        因子描述：
        24月相对强势（Relative strength for the last 24 months）。
        计算方法：

        其中N取24个自然月对应的时间长度，r表示日收益。计算中将每日无风险收益 按0 处理。
        """
        return self._RSTR(500)

    def RSTR12(self):
        """
        因子描述：
        12月相对强势（Relative strength for the last 12 months）。
        计算方法：
        其中N取12个自然月对应的时间长度，r表示日收益。计算中将每日无风险收益 按0 处理。
        """
        return self._RSTR(250)

    def TOBT(self):
        """
        因子描述：
        超额流动（Liquidity-turnover beta）。
        计算方法：
        (a) 记 为第i支股票的日收益，市场组合日收益 ， 为每日的无风险收益，则当日各自的超额日收益为 :
        和
        市场组合日收益 的计算采用沪深300 的数据。
        代码计算中将每日无风险收益 按0处理。
        (b) 每日换手率 可查询得到，也可按如下公式计算
        若流通股本数值缺失使用总股本数值代替。
        (c) 日超额收益绝对值关于换手率、市场组合日收益绝对值的五阶和自身五阶的回归表示为
        回归结果中的日换手率系数 即为所求的超额流动TOBT。
        回归窗口为过去24个月，最终结果舍去了交易日不足180 天的结果。
        其中r表示日收益，Volume表示成交量，CloseIndex表示指数今收盘，PreCloseIndex表示指数昨收盘。
        """
        rm = self.df_Index_OHLC["pct_chg"].copy()
        ri = self.df_OHLC["pct_chg"].copy()
        TORate = self.df_sb["turnover_rate_f"].copy()
        x = pd.DataFrame(rm)
        x['ri'] = ri
        x['TORate'] = TORate
        # x = pd.merge(rm, ri, on="trade_date", how='right')
        # TOBT = pd.merge(x, TORate, on="trade_date", how='left')
        for i in range(1, 6):
            x["rm" + str(i)] = x["pct_chg"].shift(i)
            x["ri" + str(i)] = x["ri"].shift(i)
        roll = ols.PandasRollingOLS(y=x["ri"].abs(), x=x[["TORate", "ri1", "ri2", "ri3", "ri4", "ri5", "rm1",
                                                                "rm2", "rm3", "rm4", "rm5"]].abs(), window=504)
        return roll.beta["TORate"]

    def PSY(self):
        """
        因子描述：
        心理线指标（Psychological line index），是一定时期内投资者趋向买方或卖方的心理事实转的数值度量，用于判断股价的未来趋势。
        计算方法：
        定义上涨日为当日收盘价高于前日收盘价，N 日内的PSY 可以表示为上涨日占分析周期的比例：
        其中I为示性函数，通常N不超过24，此处N = 12。
        """
        adj_close = self.df_OHLC['close']
        temp = adj_close.diff(1)
        temp = temp.mask(temp > 0, 1)
        temp = temp.mask(temp != 1, 0)
        PSY = temp.rolling(window=12, center=False).sum() / 12
        return PSY

    def JDQS20(self):
        """
        因子描述：
        阶段强势指标 (20-day relative strength to market)。该指标计算一定周期N日内，大盘下跌时，个股上涨的比例。
        计算方法：
        1. A = {N天中大盘收阴线，个股收阳线的天数}
        2. B = {N天中大盘收阴线的天数}。
        3. JDQS = A / B
        N = 20。
        """
        Index_open = self.df_Index_OHLC["open"]
        Index_close = self.df_Index_OHLC["close"]
        stock_open = self.df_OHLC["open"]
        stock_close = self.df_OHLC["close"]
        A = (Index_close < Index_open) & (stock_close > stock_open)
        B = (Index_close < Index_open)
        AA = A.rolling(20).sum()
        BB = B.rolling(20).sum()
        JDQS = AA / BB
        return JDQS

    def RSTR504(self):
        """
        因子描述：504天相对强势（Relative strength for the last 504 trading days）。
        计算方法：
        其中T为因子值当天，L=21，T-L代表21个交易日之前， 是股票的收益， 是无风险收益，这里用FR007代替。 为
        指数衰减权重，半衰期为126天
        """
        temp = self.df_OHLC['pct_chg']
        temp1 = (temp / 100 + 1).apply(np.log)
        temp1 = temp1.shift(21)

        def exponential(x):
            z = np.zeros(504)
            for i in range(len(z)):
                z[i] = (1 / 2) ** (i / 126)
            c = x * z
            s = c.sum()
            return s

        RSTR504 = temp1.rolling(504).apply(exponential)
        return RSTR504

