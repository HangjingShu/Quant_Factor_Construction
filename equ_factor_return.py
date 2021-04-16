import numpy as np
import pandas as pd
from pyfinance import ols
import statsmodels.api as sm

"""
get series from df_bic
calculate if necessary
drop duplicated records
expand the index list of df_bic by df_sb
div df_sb if necessary
"""


# 股票因子_收益类 35
class EquFactorReturn(Equ):
    def __init__(self, OHLC, Index_OHLC, if_qfq=False):
        super().__init__()
        self.OHLC = OHLC
        self.Index_OHLC = Index_OHLC
        self.if_qfq = if_qfq

    def Variance20(self):
        """
        因子描述：
        20日收益方差。
        计算方法：
        其中r为日收益，因子值为年化后的值，等于日度方差*250。

        """
        r = self.df_OHLC["pct_chg"]
        Variance20 = r.rolling(20, 1).var()
        Variance20_y = Variance20 * 250
        return Variance20_y

    def Variance60(self):
        """
        因子描述：
        60日收益方差。
        计算方法：
        其中r为日收益，因子值为年化后的值，等于日度方差*250。

        """
        r = self.df_OHLC["pct_chg"]
        Variance60 = r.rolling(60, 1).var()
        Variance60_y = Variance60 * 250
        return Variance60_y

    def Variance120(self):
        """
        因子描述：
        120日收益方差。
        计算方法：
        其中r为日收益，因子值为年化后的值，等于日度方差*250。

        """
        r = self.df_OHLC["pct_chg"]
        Variance120 = r.rolling(120, 1).var()
        Variance120_y = Variance120 * 250
        return Variance120_y

    def Kurtosis20(self):
        """
        因子描述：
        个股收益的20日峰度。
        计算方法：
        其中r 代表每日收益， 代表收益标准差
        """
        r = self.df_OHLC["pct_chg"]
        Kurtosis20 = r.rolling(20, 1).kurt() + 3
        return Kurtosis20

    def Kurtosis60(self):
        """
        因子描述：
        个股收益的60日峰度。
        计算方法：
        其中r 代表每日收益， 代表收益标准差
        """
        r = self.df_OHLC["pct_chg"]
        Kurtosis60 = r.rolling(60, 1).kurt() + 3
        return Kurtosis60

    def Kurtosis120(self):
        """
        因子描述：
        个股收益的120日峰度。
        计算方法：
        其中r 代表每日收益， 代表收益标准差
        """
        r = self.df_OHLC["pct_chg"]
        Kurtosis120 = r.rolling(120).kurt() + 3
        return Kurtosis120

    def _Alpha(self, date_length):
        r = self.df_OHLC["pct_chg"]
        rm = self.df_Index_OHLC['pct_chg']
        temp1 = r.rolling(window=date_length, center=False).mean()
        beta = self._Beta(date_length)
        temp2 = temp1 - beta * rm
        alpha = temp1.sub(temp2.squeeze(), axis=0) * 250
        return alpha

    def Alpha20(self):
        """
        因子描述：
        20日Jensen’s alpha。
        计算方法：
        其中：
        r 代表每日收益
        代表无风险收益，用银行间回购定盘利率FR007，下同, 代表因子值当日的FR007值 ,
        beta代表收益的20日beta值,
        注：因子值给出的是年化后的alpha20，即用求出的日度alpha*250。
        """
        return self._Alpha(20)

    def Alpha60(self):
        """

        """
        return self._Alpha(60)

    def Alpha120(self):
        """

        """
        return self._Alpha(120)

    def _Beta(self, N):
        r = self.df_OHLC["pct_chg"]
        rm = self.df_Index_OHLC['pct_chg']
        roll = ols.PandasRollingOLS(y=r, x=rm, window=N)
        return roll.beta['feature1']

    def Beta20(self):
        """
        因子描述：
        20日beta值。
        计算方法：
        将个股收益率同市场收益率进行回归，得到的回归系数 就是因子值
        说明：
        代表个股每日收益， 无风险收益率 此处为0
        代表指数收益，这里使用沪深300指数；
        回归方法为ols回归
        """
        return self._Beta(20)

    def Beta60(self):
        """
        因子描述：
        60日beta值。
        计算方法：
        将个股收益率同市场收益率进行回归，得到的回归系数 就是因子值
        说明：
        代表个股每日收益， 无风险收益率 此处为0
        代表指数收益，这里使用沪深300指数；
        回归方法为ols回归
        """
        return self._Beta(60)

    def Beta120(self):
        """
        因子描述：
        120日beta值。
        计算方法：
        将个股收益率同市场收益率进行回归，得到的回归系数 就是因子值
        说明：
        代表个股每日收益， 无风险收益率 此处为0
        代表指数收益，这里使用沪深300指数；
        回归方法为ols回归
        """
        return self._Beta(120)

    def SharpeRatio20(self):
        """
        因子描述：
        20日夏普比率，表示每承受一单位总风险，会产生多少的超额报酬，可以同时对策略的收益与风险进行综合考虑。
        计算方法：
        其中：
        E(r)代表期望收益，等于日度收益均值*250
        代表无风险收益率，使用计算日当日值，无风险收益率用FR007替代
        代表收益的标准偏差，等于日度收益标准差 *sqrt(250)
        """
        r = self.df_OHLC["pct_chg"]
        rf = 0  ## not determined
        meanr = r.rolling(20, 1).mean() * 250
        stdr = r.rolling(20, 1).std() * np.sqrt(250)
        SharpeRatio20 = (meanr - rf) / stdr
        return SharpeRatio20

    def SharpeRatio60(self):
        """
        60日夏普比率，表示每承受一单位总风险，会产生多少的超额报酬，可以同时对策略的收益与风险进行综合考虑。
        计算方法：
        其中：
        E(r)代表期望收益，等于日度收益均值*250
        代表无风险收益率，使用计算日当日值，无风险收益率用FR007替代
        代表收益的标准偏差，等于日度收益标准差 *sqrt(250)
        """
        r = self.df_OHLC["pct_chg"]
        rf = 0  ## not determined
        meanr = r.rolling(60, 1).mean() * 250
        stdr = r.rolling(60, 1).std() * np.sqrt(250)
        SharpeRatio60 = (meanr - rf) / stdr
        return SharpeRatio60

    def SharpeRatio120(self):
        """
        120日夏普比率，表示每承受一单位总风险，会产生多少的超额报酬，可以同时对策略的收益与风险进行综合考虑。
        计算方法：
        其中：
        E(r)代表期望收益，等于日度收益均值*250
        代表无风险收益率，使用计算日当日值，无风险收益率用FR007替代
        代表收益的标准偏差，等于日度收益标准差 *sqrt(250)
        """
        r = self.df_OHLC["pct_chg"]
        rf = 0  ## not determined
        meanr = r.rolling(120, 1).mean() * 250
        stdr = r.rolling(120, 1).std() * np.sqrt(250)
        SharpeRatio120 = (meanr - rf) / stdr
        return SharpeRatio120

    def TreynorRatio20(self):
        """
        因子描述：
        20日特诺雷比率，用以衡量投资回报率。
        计算方法：
        其中：
        r代表每日收益
        E(r)代表期望收益
        代表无风险收益FR007当天值
        beta代表收益的beta值, 取自因子beta20
        注：因子值是年化后的值，等于日度值* 250。
        """
        r = self.df_OHLC["pct_chg"]
        rf = 0  ## not determined
        meanr = r.rolling(20, 1).mean()
        beta20 = self.Beta20()
        TreynorRatio20 = (meanr - rf) / beta20 * 250
        return TreynorRatio20

    def TreynorRatio60(self):
        """
        因子描述：
        60日特诺雷比率，用以衡量投资回报率。
        计算方法：
        其中：
        r代表每日收益
        E(r)代表期望收益
        代表无风险收益FR007当天值
        beta代表收益的beta值, 取自因子beta20
        注：因子值是年化后的值，等于日度值* 250。
        """
        r = self.df_OHLC["pct_chg"]
        rf = 0  ## not determined
        meanr = r.rolling(60, 1).mean()
        beta60 = self.Beta60()
        TreynorRatio60 = (meanr - rf) / beta60 * 250
        return TreynorRatio60

    def TreynorRatio120(self):
        """
        因子描述：
        120日特诺雷比率，用以衡量投资回报率。
        计算方法：
        其中：
        r代表每日收益
        E(r)代表期望收益
        代表无风险收益FR007当天值
        beta代表收益的beta值, 取自因子beta20
        注：因子值是年化后的值，等于日度值* 250。
        """
        r = self.df_OHLC["pct_chg"]
        rf = 0  ## not determined
        meanr = r.rolling(120, 1).mean()
        beta120 = self._Beta(120)
        TreynorRatio120 = (meanr - rf) / beta120 * 250
        return TreynorRatio120

    def InformationRatio20(self):
        """
        因子描述：
        20日信息比率。
        计算方法：
        其中：
        r 代表每日收益
        代表指数收益，选用沪深300指数
        """
        r = self.df_OHLC["pct_chg"]
        rm = self.df_Index_OHLC['pct_chg']
        diff = r - rm
        meanr = diff.rolling(20, 1).mean()
        varr = diff.rolling(20, 1).var()
        InformationRatio20 = meanr / np.sqrt(varr)
        return InformationRatio20

    def InformationRatio60(self):
        """
        因子描述：
        60日信息比率。
        计算方法：
        其中：
        r 代表每日收益
        代表指数收益，选用沪深300指数
        """
        r = self.df_OHLC["pct_chg"]
        rm = self.df_Index_OHLC['pct_chg']
        diff = r - rm
        meanr = diff.rolling(60, 1).mean()
        varr = diff.rolling(60, 1).var()
        InformationRatio60 = meanr / np.sqrt(varr)
        return InformationRatio60

    def InformationRatio120(self):
        """
        因子描述：
        60日信息比率。
        计算方法：
        其中：
        r 代表每日收益
        代表指数收益，选用沪深300指数
        """
        r = self.df_OHLC["pct_chg"]
        rm = self.df_Index_OHLC['pct_chg']
        diff = r - rm
        meanr = diff.rolling(120, 1).mean()
        varr = diff.rolling(120, 1).var()
        InformationRatio120 = meanr / np.sqrt(varr)
        return InformationRatio120

    def GainVariance20(self):
        """
        因子描述：
        20日收益方差，类似于方差，但是主要衡量收益的表现。
        计算方法：
        其中：
        r 代表每日收益
        注：因子值是年化后的值，等于日度值*250。
        """
        r = self.df_OHLC["pct_chg"]
        r_pos = r.mask(r < 0, np.nan)
        r2 = np.square(r_pos)
        meanr2 = r2.rolling(20, 1).mean()
        mean2r = np.square(r_pos.rolling(20, 1).mean())
        GainVariance20 = (meanr2 - mean2r) * 250
        return GainVariance20

    def GainVariance60(self):
        """
        因子描述：
        60日收益方差，类似于方差，但是主要衡量收益的表现。
        计算方法：
        其中：
        r 代表每日收益
        注：因子值是年化后的值，等于日度值*250
        """
        r = self.df_OHLC["pct_chg"]
        r_pos = r.mask(r < 0, np.nan)
        r2 = np.square(r_pos)
        meanr2 = r2.rolling(60, 1).mean()
        mean2r = np.square(r_pos.rolling(60, 1).mean())
        GainVariance60 = (meanr2 - mean2r) * 250
        return GainVariance60

    def GainVariance120(self):
        """
        因子描述：
        120日收益方差，类似于方差，但是主要衡量收益的表现。
        计算方法：
        其中：
        r 代表每日收益
        注：因子值是年化后的值，等于日度值*250。
        """
        r = self.df_OHLC["pct_chg"]
        r_pos = r.mask(r < 0, np.nan)
        r2 = np.square(r_pos)
        meanr2 = r2.rolling(120, 1).mean()
        mean2r = np.square(r_pos.rolling(120, 1).mean())
        GainVariance120 = (meanr2 - mean2r) * 250
        return GainVariance120

    def LossVariance20(self):
        """
        因子描述：
        20日损失方差，类似于方差，但是主要衡量损失的表现。
        计算方法：
        其中：
        r 代表每日收益
        注：因子值是年化后的值，等于日度值*250。
        """
        r = self.df_OHLC["pct_chg"]
        r_neg = r.mask(r > 0, np.nan)
        r2 = np.square(r_neg)
        meanr2 = r2.rolling(20, 1).mean()
        mean2r = np.square(r_neg.rolling(20, 1).mean())
        GainVariance20 = (meanr2 - mean2r) * 250
        return GainVariance20

    def LossVariance60(self):
        """
        因子描述：
        60日损失方差，类似于方差，但是主要衡量损失的表现。
        计算方法：
        其中：
        r 代表每日收益
        注：因子值是年化后的值，等于日度值*250
        """
        r = self.df_OHLC["pct_chg"]
        r_neg = r.mask(r > 0, np.nan)
        r2 = np.square(r_neg)
        meanr2 = r2.rolling(60, 1).mean()
        mean2r = np.square(r_neg.rolling(60, 1).mean())
        GainVariance60 = (meanr2 - mean2r) * 250
        return GainVariance60

    def LossVariance120(self):
        """
        因子描述：
        120日损失方差，类似于方差，但是主要衡量损失的表现。
        计算方法：
        其中：
        r 代表每日收益
        注：因子值是年化后的值，等于日度值*250。
        """
        r = self.df_OHLC["pct_chg"]
        r_neg = r.mask(r > 0, np.nan)
        r2 = np.square(r_neg)
        meanr2 = r2.rolling(120, 1).mean()
        mean2r = np.square(r_neg.rolling(120, 1).mean())
        GainVariance120 = (meanr2 - mean2r) * 250
        return GainVariance120

    def GainLossVarianceRatio20(self):
        """
        因子描述：
        20日收益损失方差比。
        计算方法：
        其中：
        r 代表每日收益
        """
        GV = self.GainVariance20()
        LV = self.LossVariance20()
        GainLossVarianceRatio20 = GV / LV
        return GainLossVarianceRatio20

    def GainLossVarianceRatio60(self):
        """
        因子描述：
        60日收益损失方差比。
        计算方法：
        其中：
        r 代表每日收益
        """
        GV = self.GainVariance60()
        LV = self.LossVariance60()
        GainLossVarianceRatio60 = GV / LV
        return GainLossVarianceRatio60

    def GainLossVarianceRatio120(self):
        """
        因子描述：
        60日收益损失方差比。
        计算方法：
        其中：
        r 代表每日收益
        """
        GV = self.GainVariance120()
        LV = self.LossVariance120()
        GainLossVarianceRatio120 = GV / LV
        return GainLossVarianceRatio120

    def RealizedVolatility(self):
        """
        因子描述：
        实际波动率，日内5分钟线的收益率标准差。
        计算方法：
        使用5分钟线的close计算每5分钟的收益，然后求日内5分钟的收益的标准差
        """
        ### ???

    def Beta252(self):
        """
        计算方法：
        将个股收益率同市场收益率进行回归，得到的回归系数 就是因子值
        说明：
        代表个股每日收益， 无风险收益率 此处为7天逆回购利率FR007
        代表指数收益，这里使用沪深300指数；
        回归方法为wls回归，半衰期为63
        """
        df = pd.DataFrame()
        df['x'] = self.df_Index_OHLC['pct_chg']
        df['y'] = self.df_OHLC['pct_chg']
        df.set_index(self.df_Index_OHLC.index)
        z = np.zeros(252)
        for i in range(len(z)):
            z[i] = (1 / 2) ** (i / 63)
        Beta252 = pd.Series(np.full(len(df), np.nan), index=self.df_Index_OHLC.index)
        for j in range(int(len(df) - 251)):
            res = sm.GLS(df['y'][j:j + 252], sm.add_constant(df['x'][j:j + 252]), sigma=z).fit()
            Beta252[j] = res.params[1]
        return Beta252

    def DASTD(self):
        """
        因子描述：
        252日超额收益标准差。
        计算方法：
        第一步，计算加权算数平均数：
        第二步，计算标准差：
        """
        adj_change = self.df_OHLC['pct_chg']

        def weighted_std(x):
            z = np.zeros(252)
            for i in range(len(z)):
                z[i] = (1 / 2) ** (i / 42)
            average = np.ma.average(x, weights=z, axis=0)
            variance = np.dot(z, (x - average) ** 2) / z.sum()
            std = np.sqrt(variance)
            return std

        DASTD = adj_change.copy()
        DASTD = DASTD.rolling(252, 1).apply(weighted_std)
        return DASTD

    def CmraCNE5(self):
        """
        因子描述：
        12 月累计收益（Monthly cumulative return range over the past 12 months）。
        计算方法：
        1. 定义过去T个月的累计收益率, T=1,2, 3, .... N
        2. 计算因子值
        其中 为Z(T)的最大值， 为Z(T)的最小值
        说明：
        N = 12，无风险收益率 用7天逆回购利率FR007代替, 定义一个月为21个交易日
        """
        n = 21
        adj_close = self.df_OHLC['close']
        r = pd.DataFrame()
        for i in range(0, -12, -1):
            r_tmp = (adj_close.shift(abs(i) * n) / adj_close.shift((abs(i) + 1) * n))
            r_tmp = r_tmp.apply(np.log)
            if i == 0:
                r[abs(i)] = r_tmp
            else:
                r[abs(i)] = r[abs(i) - 1] + r_tmp
        r_max = r.max(axis=1)
        r_min = r.min(axis=1)
        CMRA = r_max - r_min
        return CMRA

    def HsigmaCNE5(self):
        """
        因子描述：
        252日残差收益波动率。
        计算方法：
        将个股收益率同市场收益率进行加权回归，并得到残差的波动率
        说明：
        代表个股每日收益， 无风险收益率 此处为7天逆回购利率
        代表指数收益，这里使用沪深300指数；
        回归方法为wls回归，半衰期为63，f为半衰期对应的权重
        """
        df = pd.DataFrame()
        df['x'] = self.df_Index_OHLC['pct_chg']
        df['y'] = self.df_OHLC['pct_chg']
        df.set_index(self.df_Index_OHLC.index)
        z = np.zeros(30)
        for i in range(len(z)):
            z[i] = (1 / 2) ** (i / 30)
        HsigmaCNE5 = pd.Series(np.full(len(df), np.nan), index=self.df_Index_OHLC.index)
        for j in range(int(len(df) - 29)):
            res = sm.GLS(df['y'][j:j + 30], sm.add_constant(df['x'][j:j + 30]), sigma=z).fit()
            resid_mean = (res.resid * z) / z.sum()
            c = (res.resid - resid_mean).apply(np.square)
            d = (c * z).sum() / z.sum()
            d = d ** (1 / 2)
            HsigmaCNE5[j] = d
        return HsigmaCNE5


